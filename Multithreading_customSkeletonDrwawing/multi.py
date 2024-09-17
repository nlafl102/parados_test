import multiprocessing
import threading
import time
import ffmpeg
import os
import mediapipe as mp
import cv2
import subprocess
import sys
import shutil
import csv
import matplotlib.pyplot as plt
import numpy as np
from functools import partial


from landmark_config import LandmarkStyleConfig

# comment


res = "Result-Skeleton"


def install(package):
    try:
        subprocess.check_call(
            [sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        print(f"Failed to install {package}.")
        return False


class MediaPipeConfig:
    def __init__(
        self,
        line_thickness=2,
        landmark_styles={},
        excluded_connections=[],
        specific_connections=[],
    ):
        self.line_thickness = line_thickness
        self.landmark_styles = landmark_styles
        self.specific_connections = specific_connections
        self.excluded_connections = excluded_connections


class VideoProcessor:
    def __init__(self, video_path, mediapipe_config, frame_range):
        self.video_path = video_path
        self.frame_range = frame_range
        self.output_folder = f"./frames-{res}"
        self.cap = cv2.VideoCapture(self.video_path)
        self.pose = mp.solutions.pose.Pose()
        self.mediapipe_config = mediapipe_config
        self.body_coordinates = []
        self.lock = threading.Lock()

    def resize_frame(self, frame, max_width=1080, max_height=1920):
        aspect_ratio = frame.shape[1] / frame.shape[0]
        if frame.shape[0] > max_height:
            new_height = max_height
            new_width = int(new_height * aspect_ratio)
            frame = cv2.resize(frame, (new_width, new_height))
        return frame

    def process_frame(self, frame_counter, frame):
        frame = self.resize_frame(frame)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.pose.process(frame_rgb)

        print(".", end='', flush=True)

        if results.pose_landmarks:
            mp_drawing = mp.solutions.drawing_utils
            frame_with_pose = frame.copy()

            landmarks = results.pose_landmarks.landmark
            landmark_data = {
                "frame_number": frame_counter,
                "coordinates": [
                    {
                        "landmark": i,
                        "x": lm.x,
                        "y": lm.y,
                        "z": lm.z,
                        "visibility": lm.visibility
                    }
                    for i, lm in enumerate(landmarks)
                ]
            }
            self.body_coordinates.append(landmark_data)
            # Create custom DrawingSpec for each landmark
            custom_landmark_style_dict = {
                i: mp_drawing.DrawingSpec(
                    color=self.mediapipe_config.landmark_styles[i]["color"],
                    thickness=self.mediapipe_config.landmark_styles[i]["thickness"],
                    circle_radius=self.mediapipe_config.landmark_styles[i][
                        "circle_radius"
                    ],
                )
                for i in self.mediapipe_config.landmark_styles
            }

            # Dynamically determine connections to include
            connections = [
                conn
                for conn in mp.solutions.pose.POSE_CONNECTIONS
                if self.mediapipe_config.landmark_styles[conn[0]]["thickness"]
                is not None
                and self.mediapipe_config.landmark_styles[conn[1]]["thickness"]
                is not None
            ]

            # Explicitly add specific connections
            connections.extend(self.mediapipe_config.specific_connections)

            # Draw landmarks and connections with custom styles
            mp_drawing.draw_landmarks(
                frame_with_pose,
                results.pose_landmarks,
                connections,  # Existing connections
                landmark_drawing_spec=custom_landmark_style_dict,
                connection_drawing_spec=mp_drawing.DrawingSpec(
                    thickness=self.mediapipe_config.line_thickness,
                ),
            )

            frame_filename = os.path.join(
                self.output_folder, f"frame_{frame_counter:04d}.jpg"
            )

            cv2.imwrite(frame_filename, frame_with_pose)

        else:
            frame_filename = os.path.join(
                self.output_folder, f"frame_{frame_counter:04d}.jpg"
            )
            cv2.imwrite(frame_filename, frame)

    def export_to_csv(self, output_csv_path):
        """
        Appends the collected body coordinates to a CSV file in a thread-safe manner.
        """
        with self.lock:  # Ensure thread-safe file access
            with open(output_csv_path, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Write the data for each frame
                for frame_data in self.body_coordinates:
                    frame_number = frame_data["frame_number"]
                    for coord in frame_data["coordinates"]:
                        writer.writerow([
                            frame_number,
                            coord["landmark"],
                            coord["x"],
                            coord["y"],
                            coord["z"],
                            coord["visibility"]
                        ])
        print(f"Data successfully appended to {output_csv_path}")

    def process_video_segment(self):
        frame_counter = self.frame_range[0]
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_counter)

        while frame_counter <= self.frame_range[1]:
            ret, frame = self.cap.read()
            if not ret:
                break

            self.process_frame(frame_counter, frame)
            frame_counter += 1

        self.cap.release()
        cv2.destroyAllWindows()
        return self.output_folder


class VideoAssembler:
    def __init__(self, frame_folder):
        self.frame_folder = frame_folder

    def assemble_video(self):
        if not os.path.exists(self.frame_folder):
            print("Frame folder not found.")
            return

        frame_files = sorted(
            [f for f in os.listdir(self.frame_folder) if f.endswith(".jpg")]
        )
        if not frame_files:
            print("No frame files found in the folder.")
            return

        output_path = f"final_output-{os.path.basename(self.frame_folder)}-27.mp4"

        input_stream = ffmpeg.input(
            os.path.join(self.frame_folder, "frame_%04d.jpg"), framerate=30
        )

        num_threads = multiprocessing.cpu_count()  # Get the number of available threads

        print('\n Threads: '+str(num_threads))

        output_stream = ffmpeg.output(
            input_stream,
            output_path,
            **{"codec:v": "libx264"},
            vf="pad=ceil(iw/2)*2:ceil(ih/2)*2",  # Pad to even dimensions
            preset="veryfast",
            crf=27,
            threads=num_threads,
        )

        # overwrite = ffmpeg.overwrite_output(output_stream)

        ffmpeg.run(output_stream, quiet=False, overwrite_output=True)

        # input_stream = input(input_path, framerate=self.frame_rate)
        # output_stream = output(
        #     input_stream, output_path, **{"codec:v": self.codec}, preset="fast", crf=24, threads=num_threads
        # )


# def get_user_selection_inquirer(prompt, options):
#     questions = [
#         inquirer.List(
#             "select",
#             message=prompt,
#             choices=options,
#         ),
#     ]
#     answers = inquirer.prompt(questions)
#     return answers["select"]

def replace_drawing_utils_if_needed():
    # Update with the correct path
    custom_file_path = 'drawing_utils/custom_drawing_utils.py'
    mp_file_path = os.path.join(
        os.path.dirname(mp.__file__), 'python', 'solutions', 'drawing_utils.py')
    # Check if the first line of the existing file matches the specified comment
    try:
        with open(mp_file_path, 'r') as file:
            first_line = file.readline()
            if '# modified by Parados' not in first_line:
                shutil.copyfile(custom_file_path, mp_file_path)
                print("MediaPipe drawing_utils.py replaced with custom version.")
            else:
                print("Custom MediaPipe drawing_utils.py already in place.")
    except IOError as e:
        print(
            f"Error occurred while checking or replacing drawing_utils.py: {e}")


def get_video_resolution(video_path):
    cap = cv2.VideoCapture(video_path)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cap.release()
    return width, height

def split_video(video_path, num_threads):
    cap = cv2.VideoCapture(video_path)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frames_per_thread = frame_count // num_threads
    frame_ranges = []

    for i in range(num_threads):
        start_frame = i * frames_per_thread
        end_frame = start_frame + frames_per_thread - 1
        frame_ranges.append((start_frame, end_frame))

    # Make sure the last thread processes any remaining frames
    frame_ranges[-1] = (frame_ranges[-1][0], frame_count - 1)

    return frame_ranges

def process_video_segment(args):
    video_path, frame_range, mediapipe_config, csv_path = args
    video_processor = VideoProcessor(video_path, mediapipe_config, frame_range)
    video_processor.process_video_segment()
    video_processor.export_to_csv(csv_path)

def sort_csv_by_columns(csv_path, columns=[0, 1]):
    """
    Sorts the rows of a CSV file by multiple columns and writes the sorted rows back to the same file.
    
    :param csv_path: Path to the CSV file.
    :param columns: List of column indices to sort by (0-based). Default is [0, 1] (first and second columns).
    """
    # Read the CSV file
    with open(csv_path, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Read the header
        rows = list(reader)    # Read all rows into a list

    # Sort rows based on multiple column indices
    rows.sort(key=lambda row: tuple(float(row[col]) for col in columns))  # Convert values to float for numerical sorting

    # Write the sorted data back to the same file
    with open(csv_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header)  # Write the header
        writer.writerows(rows)   # Write the sorted rows

def calculate_varus_angles_over_frames(csv_path):
    """
    Calculates the varus angle for both legs across all frames and stores the varus angles in the CSV file.
    
    :param csv_path: Path to the CSV file.
    :return: Two lists containing varus angles for left and right legs along with the frames.
    """
    markers_of_interest = {
        'left_hip': 23,
        'left_knee': 25,
        'left_ankle': 27,
        'right_hip': 24,
        'right_knee': 26,
        'right_ankle': 28
    }
    
    varus_angles_left = []
    varus_angles_right = []
    frames = []
    coordinates_per_frame = {}
    
    # Read the CSV file and group coordinates by frame
    with open(csv_path, mode='r', newline='') as infile:
        reader = csv.reader(infile)
        header = next(reader)  # Skip header
        
        data = []
        for row in reader:
            try:
                frame_num = int(row[0])
                marker_id = int(row[1])
                x = float(row[2])
                y = float(row[3])
                
                if frame_num not in coordinates_per_frame:
                    coordinates_per_frame[frame_num] = {}
                
                if marker_id in markers_of_interest.values():
                    coordinates_per_frame[frame_num][marker_id] = (x, y)
                
                data.append(row)  # Save original data
                
            except ValueError as e:
                print(f"Skipping row due to conversion error: {e}, row: {row}")
    
    # Calculate varus angles for each frame
    for frame_number, coordinates in coordinates_per_frame.items():
        if all(marker in coordinates for marker in markers_of_interest.values()):
            # Get coordinates for the markers
            left_hip = np.array(coordinates[markers_of_interest['left_hip']])
            left_knee = np.array(coordinates[markers_of_interest['left_knee']])
            left_ankle = np.array(coordinates[markers_of_interest['left_ankle']])
            
            right_hip = np.array(coordinates[markers_of_interest['right_hip']])
            right_knee = np.array(coordinates[markers_of_interest['right_knee']])
            right_ankle = np.array(coordinates[markers_of_interest['right_ankle']])
            
            # Create vectors for thigh and shin
            left_thigh_vector = left_knee - left_hip
            left_shin_vector = left_ankle - left_knee
            
            right_thigh_vector = right_knee - right_hip
            right_shin_vector = right_ankle - right_knee
            
            # Calculate varus angles
            left_varus_angle = calculate_signed_angle(left_thigh_vector, left_shin_vector)
            right_varus_angle = calculate_signed_angle(right_thigh_vector, right_shin_vector)
            
            # Append to lists
            frames.append(frame_number)
            varus_angles_left.append(-1 * left_varus_angle)
            varus_angles_right.append(right_varus_angle)
    
    # Add new data to the original CSV file
    new_csv_data = []
    for row in data:
        try:
            frame_num = int(row[0])
            if frame_num in frames:
                # Get the corresponding varus angles
                left_angle = varus_angles_left[frames.index(frame_num)]
                right_angle = varus_angles_right[frames.index(frame_num)]
                # Append the angles to the row
                row.extend([left_angle, right_angle])
            else:
                row.extend([None, None])  # If no angles were calculated for this frame
        except ValueError:
            row.extend([None, None])  # In case of header or invalid rows
        
        new_csv_data.append(row)
    
    # Write the updated data back to the CSV file (with varus angle columns)
    with open(csv_path, mode='w', newline='') as outfile:
        writer = csv.writer(outfile)
        writer.writerow(header + ['Left Varus Angle', 'Right Varus Angle'])  # Updated header
        writer.writerows(new_csv_data)
    
    # Return the frames and angles for plotting
    return frames, varus_angles_left, varus_angles_right

def plot_varus_angles(frames, varus_angles_left, varus_angles_right):
    """
    Plots the varus angles for both legs as a function of frame number.
    
    :param frames: List of frame numbers.
    :param varus_angles_left: List of varus angles for the left leg.
    :param varus_angles_right: List of varus angles for the right leg.
    """
    # Plot the varus angles as a function of frame number
    plt.figure(figsize=(12, 6))
    frames = np.array(frames)
    varus_angles_left = np.array(varus_angles_left)
    varus_angles_right = np.array(varus_angles_right)
    plt.plot(frames, varus_angles_left, label='Left Leg', color='blue', marker='o')
    plt.plot(frames, varus_angles_right, label='Right Leg', color='red', marker='o')
    plt.axhline(y=0, color='black', linestyle='-', label='Neutral (0 degrees)')
    # Fill areas based on angle categories
    plt.fill_between(frames, varus_angles_left, 0, where=(varus_angles_left < 0), color='orange', alpha=0.3, label='Varus Angle')
    plt.fill_between(frames, varus_angles_left, 0, where=(varus_angles_left > 0), color='green', alpha=0.3, label='Valgus Angle')
    plt.fill_between(frames, varus_angles_right, 0, where=(varus_angles_right < 0), color='orange', alpha=0.3)
    plt.fill_between(frames, varus_angles_right, 0, where=(varus_angles_right > 0), color='green', alpha=0.3)
    plt.title("Valgus Angle vs Frames for Both Legs")
    plt.xlabel("Frame Number")
    plt.ylabel("Valgus Angle (degrees)")
    plt.legend()
    plt.grid(True)
    plt.show()

def calculate_signed_angle(v1, v2):
    """
    Calculates the signed angle between two 2D vectors.
    :param v1: First vector.
    :param v2: Second vector.
    :return: Signed angle in degrees.
    """
    angle = np.degrees(np.arctan2(np.linalg.det([v1, v2]), np.dot(v1, v2)))
    return angle


def main():
    # Define options
    view_options = ["Front", "Left", "Right", "Back"]
    category_options = [
        "Upper Body Exercises",
        "Lower Body Exercises",
        "Rehabilitation Exercises",
        "Functional Movements",
        "Shooting",
    ]

    # # Get user selection through interactive CLI
    # selected_view = get_user_selection_inquirer("Select View", view_options)
    # selected_category = get_user_selection_inquirer(
    #     "Select Exercise Category", category_options
    # )

    # Default Selection:
    selected_view = 'Front'
    selected_category = 'Upper Body Exercises'

    video_path = './videos/squat1080.mp4'
    csv_path = 'squat1080.csv'
    with open(csv_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            # Write CSV header
            writer.writerow(["Frame", "Landmark", "X", "Y", "Z", "Visibility"])
    output_folder = os.path.join(".", f"frames-{res}")
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    else:
        print(f"Folder '{output_folder}' already exists, skipping creation.")

    # Get video resolution
    replace_drawing_utils_if_needed()

    video_width, video_height = get_video_resolution(video_path)
    scale = 1

    # Calculate the scale based on video resolution
    if video_width <= 1080:
        scale = video_width / 1080  # You can adjust this value as needed

    style_config = LandmarkStyleConfig(selected_view, selected_category, scale)

    mediapipe_config = MediaPipeConfig(
        line_thickness=style_config.global_line_thickness,
        landmark_styles=style_config.landmark_styles,
        excluded_connections=style_config.excluded_connections,
        specific_connections=style_config.specific_connections,
    )

    # Measure the time for VideoAssembler
    start_time_video_processor = time.time()

    num_threads = multiprocessing.cpu_count()

    # Split the video into segments
    frame_ranges = split_video(video_path, num_threads)

    # Create a list of arguments for each segment
    process_args = [(video_path, frame_range, mediapipe_config, csv_path)
                    for frame_range in frame_ranges]

    # Initialize a multiprocessing pool
    with multiprocessing.Pool(len(frame_ranges)) as pool:
        # Process video segments in parallel
        pool.map(process_video_segment, process_args)

    # Wait for all processes to finish
    pool.close()
    pool.join()  # This line blocks until all processes are done

    end_time_video_processor = time.time()
    # After processing video segments, get the frame_folder path
    frame_folder = os.path.join(".", f"frames-{res}")

    # Measure the time for VideoAssembler
    start_time_video_assembler = time.time()
    video_assembler = VideoAssembler(frame_folder)
    video_assembler.assemble_video()
    end_time_video_assembler = time.time()

    # Calculate execution times
    execution_time_video_processor = end_time_video_processor - start_time_video_processor
    execution_time_video_assembler = end_time_video_assembler - start_time_video_assembler
    total_execution_time = end_time_video_assembler - start_time_video_processor

    # Display execution times
    print(
        f"Processing by MediaPipe took {execution_time_video_processor} seconds")
    print(
        f"FFMPEG assembled the frames into a final video in {execution_time_video_assembler} seconds")
    print(f"Total script execution time: {total_execution_time} seconds")
    sort_csv_by_columns(csv_path, columns=[0, 1])
    frames, varus_angles_left, varus_angles_right = calculate_varus_angles_over_frames(csv_path)
    plot_varus_angles(frames, varus_angles_left, varus_angles_right)

if __name__ == "__main__":
    # try:
    #     import inquirer
    # except ImportError:
    #     print("Inquirer not found. Installing inquirer...")
    #     if install("inquirer"):
    #         print("Restarting script to load the newly installed package...")
    #         os.execl(sys.executable, sys.executable, *sys.argv)
    #     else:
    #         sys.exit(
    #             "Could not install required packages. Please install them manually and rerun the script."
    #         )
    main()
