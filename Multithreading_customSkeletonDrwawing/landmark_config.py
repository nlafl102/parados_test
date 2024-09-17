import mediapipe as mp
import math


class LandmarkStyleConfig:
    def __init__(self, view, exercise_category, scale=1):
        self.view = view
        self.exercise_category = exercise_category

        # print('scale = ' + str(scale))

        self.global_dot_thickness = math.ceil(4 * scale)
        self.global_circle_radius = math.ceil(3 * scale)
        self.global_circle_radius_enlarged = math.ceil(3 * scale)
        self.global_line_thickness = math.ceil(6 * scale)

        # Default colors
        self.blue_color = (195, 137, 12)  # BGR for Hex #0C89C3
        self.orange_color = (81, 171, 245)  # BGR for Hex #F5AB51
        self.white_color = (224, 224, 224)  # offWhite in BGR

        self.landmark_styles = self.initialize_landmark_styles()
        self.excluded_connections = self.initialize_excluded_connections()
        self.specific_connections = self.initialize_specific_connections()

    def initialize_landmark_styles(self):
        # Initialize landmark styles using a dictionary
        landmark_styles = {
            i: {
                "color": self.blue_color if i % 2 != 0 else self.orange_color,
                "thickness": self.global_dot_thickness,
                "circle_radius": self.global_circle_radius
            } for i in range(33)
        }

        # Define changes for specific landmarks based on exercise category
        landmark_changes = {
            'Lower Body Exercises': [
                (mp.solutions.pose.PoseLandmark.LEFT_KNEE.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.LEFT_HIP.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.LEFT_ANKLE.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_ANKLE.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                # Add other changes for Lower Body Exercises
            ],
            'Upper Body Exercises': [
                (mp.solutions.pose.PoseLandmark.LEFT_WRIST.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_WRIST.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                # Add other changes for Upper Body Exercises
            ],
            'Shooting': [
                (mp.solutions.pose.PoseLandmark.LEFT_ELBOW.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.LEFT_HIP.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                # Add other changes for Shooting
            ],
            'Functional Movements': [
                (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.LEFT_HIP.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.LEFT_KNEE.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                # Add other changes for Functional Movements
            ],
            'Rehabilitation Exercises': [
                (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.LEFT_HIP.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_HIP.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.LEFT_KNEE.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                (mp.solutions.pose.PoseLandmark.RIGHT_KNEE.value, {
                 "circle_radius": self.global_circle_radius_enlarged}),
                # Add other changes for Rehabilitation Exercises
            ],
            # Add more exercise categories and their landmark changes as needed
        }

        # Apply changes for the specific exercise category
        if self.exercise_category in landmark_changes:
            for landmark_id, changes in landmark_changes[self.exercise_category]:
                landmark_styles[landmark_id].update(changes)

         # Get the excluded landmarks for the selected view
        excluded_landmarks = self.get_excluded_landmarks_based_on_view()

        # Update landmark styles for excluded landmarks
        for landmark, style_changes in excluded_landmarks:
            landmark_styles[landmark]["thickness"] = style_changes.get(
                "thickness", None)

        # Set the nose color specifically to white for the Front view
        # if self.view != 'Back':
        #     landmark_styles[mp.solutions.pose.PoseLandmark.NOSE.value] = {
        #         "color": self.white_color,
        #         "thickness": self.global_dot_thickness,
        #         "circle_radius": self.global_circle_radius
        #     }

        return landmark_styles

    def get_excluded_landmarks_based_on_view(self):

        initial_excluded_landmarks = [
            (mp.solutions.pose.PoseLandmark.LEFT_EYE_OUTER,
             {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.LEFT_EYE, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.RIGHT_EYE_OUTER,
             {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.RIGHT_EYE, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.LEFT_EAR, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.RIGHT_EAR, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.MOUTH_LEFT, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.MOUTH_RIGHT, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.LEFT_PINKY, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.RIGHT_PINKY, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.LEFT_INDEX, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.RIGHT_INDEX, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.LEFT_THUMB, {"thickness": None}),
            (mp.solutions.pose.PoseLandmark.RIGHT_THUMB, {"thickness": None}),

        ]

        view_to_excluded_landmarks = {
            'Front': initial_excluded_landmarks + [
                (mp.solutions.pose.PoseLandmark.NOSE, {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_HEEL,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
                 {"thickness": None}),
            ],
            'Left': initial_excluded_landmarks + [
                (mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_ELBOW,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_WRIST,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_HIP,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_KNEE,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_ANKLE,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
                 {"thickness": None}),
                # Add Left view specific landmarks here
            ],
            'Right': initial_excluded_landmarks + [
                (mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_SHOULDER,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_ELBOW,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_WRIST,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_HIP, {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_KNEE,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_ANKLE,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_HEEL,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
                 {"thickness": None}),
                # Add Right view specific landmarks here
            ],
            'Back': initial_excluded_landmarks + [
                (mp.solutions.pose.PoseLandmark.NOSE, {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_HEEL,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_HEEL,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.LEFT_FOOT_INDEX,
                 {"thickness": None}),
                (mp.solutions.pose.PoseLandmark.RIGHT_FOOT_INDEX,
                 {"thickness": None}),
                # Add Back view specific landmarks here
            ],
        }

        return view_to_excluded_landmarks.get(self.view, [])

    def initialize_excluded_connections(self):
        excluded_connections = [
            connection
            for connection in mp.solutions.pose.POSE_CONNECTIONS
            if self.landmark_styles[connection[0]]["thickness"] is None or
            self.landmark_styles[connection[1]]["thickness"] is None
        ]

        return excluded_connections  # Add this line to return the list

    def initialize_specific_connections(self):
        # Define specific connections to include
        specific_connections = []

        # # Include nose to left eye inner connection if left eye inner is visible
        # if self.landmark_styles[mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER]["thickness"] is not None:
        #     specific_connections.append(
        #         (mp.solutions.pose.PoseLandmark.NOSE,
        #          mp.solutions.pose.PoseLandmark.LEFT_EYE_INNER)
        #     )

        # # Include nose to right eye inner connection if right eye inner is visible
        # if self.landmark_styles[mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER]["thickness"] is not None:
        #     specific_connections.append(
        #         (mp.solutions.pose.PoseLandmark.NOSE,
        #          mp.solutions.pose.PoseLandmark.RIGHT_EYE_INNER)
        #     )

        return specific_connections  # Add this line to return the list
