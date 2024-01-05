import math
import os

import cv2
import mediapipe as mp
import numpy as np
import requests


def downloadImage(url, name):
    response = requests.get(url)
    if response.status_code == 200:
        with open(name, 'wb') as file:
            file.write(response.content)
        return True
    else:
        return False


def process_image(name):
    img = cv2.imread(name, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (400, 800))
    img1 = img

    # mediapipe code
    mp_drawing = mp.solutions.drawing_utils
    mp_pose = mp.solutions.pose

    with mp_pose.Pose(min_tracking_confidence=0.8, min_detection_confidence=0.5) as pose:
        img.flags.writeable = False
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = pose.process(img)
        img.flags.writeable = True
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

        mp_drawing.draw_landmarks(
            img,
            result.pose_landmarks,
            mp_pose.POSE_CONNECTIONS
        )

        # Extracting Points
        try:
            landmarks = result.pose_landmarks.landmark

        except:
            pass

    threshold_value = 128

    _, binary_image = cv2.threshold(img1, threshold_value, 255, cv2.THRESH_BINARY)
    _, binary_image1 = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    def find_x_y(landmark_index):
        if result.pose_landmarks:
            chosen_landmark = result.pose_landmarks.landmark[landmark_index]
            x, y = chosen_landmark.x, chosen_landmark.y
            x = round(x * 400)
            y = round(y * 800)
            return x, y

    # Height Calculation

    def find_pixel_vertically(binary_image, start_row, start_col, target_value):
        """Find the coordinates of a target pixel vertically from the starting point."""
        # height, width = binary_image.shape

        for row in range(start_row, -1, -1):  # Decrement the y-axis value
            if binary_image[row, start_col] == target_value:
                return row, start_col

        # Return None if the target value is not found
        return None

    # Define a kernel for dilation and erosion
    kernel = np.ones((3, 3), np.uint8)

    # Apply erosion
    eroded_image = cv2.erode(binary_image, kernel, iterations=13)

    # Apply dilation
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=20)

    RHeely, RHeelx = find_x_y(30)

    headcoordx, headcoordy = find_pixel_vertically(dilated_image, 400, 200, 255)

    x1, y1 = headcoordx, headcoordy
    x2, y2 = RHeelx, 400
    height = round(math.sqrt((RHeelx - headcoordx) ** 2 + (400 - 400) ** 2))

    LSy, LSx = find_x_y(11)
    RSy, RSx = find_x_y(12)
    LHy, LHx = find_x_y(24)
    RHy, RHx = find_x_y(23)

    LHx = LHx - 20
    RHx = RHx - 20

    def edge_dectection(binary_image, start_row, start_col, target_value, search):
        """Find the coordinates of a target pixel horizontally from the starting point."""
        height, width = binary_image.shape
        if search == "L":
            for col in range(start_col, -1, -1):
                if binary_image[start_row, col] == target_value:
                    return start_row, col
            return None
        elif search == "R":
            for col in range(start_col, width):
                if binary_image[start_row, col] == target_value:
                    return start_row, col
            return None

    # Find the coordinates of the target pixel horizontally
    def finddist(binary_image, Lx, Ly, Rx, Ry):
        result_L = edge_dectection(binary_image, Lx, Ly, 255, "L")
        result_R = edge_dectection(binary_image, Rx, Ry, 255, "R")

        x1, y1 = result_L
        x2, y2 = result_R
        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distance = round(distance)
        return distance

    shoulder = finddist(binary_image, LSx, LSy, RSx, RSy)
    waist = (finddist(binary_image, LHx, LHy, RHx, RHy)) * 2

    # Working on waist
    min_waist = 0.45 * height
    max_waist = 0.48 * height

    if min_waist <= waist <= max_waist:
        res_waist_coordinate = str(waist)
        res_waist_message = "Waist is Good"
        demowaist = waist

    else:
        res_waist_coordinate = str(waist)
        res_waist_message = "Waist is Good"
        demowaist = round((min_waist + max_waist) / 2)

    # Working on Shoulder length
    shoulder = shoulder * 2
    expected_shoulder = round(demowaist * 1.618)
    delete_files_except(["app.py", "main.py", "ResponseBody.py", "serviceAccountKey.json"], ".")

    # Check if the shoulder size is within the acceptable range
    if abs(shoulder - expected_shoulder) <= 10:
        res_shoulder_coordinate = str(shoulder)
        res_shoulder_message = "Shoulders are Good"
        return res_waist_coordinate, res_waist_message, res_shoulder_coordinate, res_shoulder_message

    else:
        res_shoulder_coordinate = str(shoulder)
        res_shoulder_message = "Shoulders needs to be worked on"
        return res_waist_coordinate, res_waist_message, res_shoulder_coordinate, res_shoulder_message


def delete_files_except(files_to_keep, root_directory):
    try:
        # List all files in the root directory
        all_files = os.listdir(root_directory)

        # Iterate through all files
        for file_name in all_files:
            file_path = os.path.join(root_directory, file_name)

            # Check if the file is in the list of files to keep
            if file_name not in files_to_keep:
                # Delete the file
                os.remove(file_path)
                print(f"Deleted: {file_path}")

        print("Deletion completed.")
    except Exception as e:
        print(f"An error occurred: {e}")
