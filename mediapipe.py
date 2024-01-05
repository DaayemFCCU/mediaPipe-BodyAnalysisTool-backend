import os
import cv2
import numpy as np
import mediapipe as mp
import math


def mediapipe(name):
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
            mp_pose.POSE_CONNECTIONS,
            mp_drawing.DrawingSpec(color=(245, 117, 66), thickness=2, circle_radius=2),
            mp_drawing.DrawingSpec(color=(245, 66, 230), thickness=2, circle_radius=2),
        )

        # Extracting Points
        try:
            landmarks = result.pose_landmarks.landmark

        except:
            pass

    threshold_value = 128

    _, binary_image = cv2.threshold(img1, threshold_value, 255, cv2.THRESH_BINARY)
    _, binary_image1 = cv2.threshold(img, threshold_value, 255, cv2.THRESH_BINARY)

    def findxyz(landmark_index):
        if result.pose_landmarks:
            chosen_landmark = result.pose_landmarks.landmark[landmark_index]
            x, y = chosen_landmark.x, chosen_landmark.y
            x = round(x * 400)
            y = round(y * 800)
            return (x, y)

    LSy, LSx = findxyz(11)
    RSy, RSx = findxyz(12)
    LHy, LHx = findxyz(24)
    RHy, RHx = findxyz(23)
    RHeely, RHeelx = findxyz(30)
    Lelbowy, Lelbowx = findxyz(13)
    Relbowy, Relbowx = findxyz(14)
    R_wristy, R_wristx = findxyz(16)
    L_wristy, L_wristx = findxyz(15)

    return (
    binary_image, binary_image1, LSx, LSy, RSx, RSy, LHx, LHy, RHx, RHy, RHeelx, Lelbowx, Lelbowy, Relbowx, Relbowy,
    R_wristx, R_wristy, L_wristx, L_wristy)


def postpross(pic1, pic2):
    binary_image, binary_image1, LSx, LSy, RSx, RSy, LHx, LHy, RHx, RHy, RHeelx, left_elbowx, left_elbowy, right_elbowx, right_elbowy, R_wristx, R_wristy, L_wristx, L_wristy = mediapipe(
        pic1)
    Simg, Simg1, SLSx, SLSy, SRSx, SRSy, SLHx, SLHy, SRHx, SRHy, SRHeelx, Sleft_elbowx, Sleft_elbowy, Sright_elbowx, Sright_elbowy, SR_wristx, SR_wristy, SL_wristx, SL_wristy = mediapipe(
        pic2)

    def find_pixel_vertically(binary_image, start_row, start_col, target_value):
        """Find the coordinates of a target pixel vertically from the starting point."""

        for row in range(start_row, -1, -1):  # Decrement the y-axis value
            if binary_image[row, start_col] == target_value:
                return row, start_col

        # Return None if the target value is not found
        return None

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

    def finddistside(binary_image, Lx, Ly, Rx, Ry):
        result_L = edge_dectection(binary_image, Lx, Ly, 0, "L")
        result_R = edge_dectection(binary_image, Rx, Ry, 0, "R")

        x1, y1 = result_L
        x2, y2 = result_R

        distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
        distance = round(distance)
        return distance

    # Height Calculation
    # Define a kernel for dilation and erosion
    kernel = np.ones((3, 3), np.uint8)

    # Apply erosion
    eroded_image = cv2.erode(binary_image, kernel, iterations=13)

    # Apply dilation
    dilated_image = cv2.dilate(eroded_image, kernel, iterations=20)

    headcoordx, headcoordy = find_pixel_vertically(dilated_image, 400, 200, 255)

    x1, y1 = headcoordx, headcoordy
    x2, y2 = RHeelx, 400
    height = round(math.sqrt((RHeelx - headcoordx) ** 2 + (400 - 400) ** 2))
    LHx = LHx - 35
    RHx = RHx - 35

    shoulder = (finddist(binary_image, LSx, LSy, RSx, RSy)) * 2
    waist = (finddist(binary_image, LHx, LHy, RHx, RHy)) * 2

    # Working on waist
    min_waist = 0.45 * height
    max_waist = 0.50 * height
    demowaist = 0.48 * height

    res_waist_coordinate = str(waist)

    if min_waist <= waist <= max_waist:
        res_waist_message = "Waist is Good."

    elif waist > max_waist:
        res_waist_message = "Waist is Big."

    else:
        res_waist_message = "Waist is Small"

    # Working on Shoulder length
    expected_shoulder = round(waist * 1.618)
    min_shldr = expected_shoulder - 20
    max_shldr = expected_shoulder + 20
    res_shoulder_coordinate = str(shoulder)

    # Check if the shoulder size is within the acceptable range

    if min_shldr <= shoulder <= max_shldr:
        res_shoulder_message = "Shoulders are Good."


    elif shoulder > max_shldr:
        res_shoulder_message = "Shoulders are Big."

    else:
        res_shoulder_message = "Shoulders are Small."

    # Working on Biceps
    # Finding Middle point between Shoulder and Elbow

    def armprocess(Sx, Sy, elbowx, elbowy, string):

        arm_midx = int(abs((Sx + elbowx) / 2))
        arm_midy = int(abs((Sy + elbowy) / 2))
        arm_midx = (int(abs((arm_midx + elbowx) / 2))) - 10
        arm_midy = int(abs((arm_midy + elbowy) / 2))

        if string == "front":
            arm = (finddist(binary_image, arm_midx, arm_midy, arm_midx, arm_midy))
            return (arm)

        elif string == "side":
            arm = (finddistside(Simg, arm_midx, arm_midy, arm_midx, arm_midy))
            return (arm)

    # Wrist calculations
    def wrist(wristx, wristy):
        wrist = int(abs(finddist(binary_image, wristx, wristy, wristx, wristy) * 2))
        return (wrist)

    left_wrist = wrist(L_wristx, L_wristy)
    right_wrist = wrist(R_wristx, R_wristy)
    wristsize = int(abs((left_wrist + right_wrist) / 2))
    flexedbicep = int(wristsize * 2.5)

    left_arm = armprocess(LSx, LSy, left_elbowx, left_elbowy, "front")

    right_arm = armprocess(RSx, RSy, right_elbowx, right_elbowy, "front")

    total_left_arm = int(left_arm * math.pi)
    total_right_arm = int(right_arm * math.pi)

    # flex_arm Calculations
    total_left_arm = int(total_left_arm * 1.05)
    total_right_arm = int(total_left_arm * 1.05)

    min_arm = flexedbicep - 10
    max_arm = flexedbicep + 10

    # Comparisons
    if min_arm <= total_left_arm <= max_arm:
        left_arm_msg = "Left Arm is Good"

    elif total_left_arm < min_arm:
        left_arm_msg = "Left Arm is Small"

    else:
        left_arm_msg = "Left Arm is Big"

    if min_arm <= total_right_arm <= max_arm:
        right_arm_msg = "Right Arm is Good"


    elif total_right_arm < min_arm:
        right_arm_msg = "Right Arm is Small"


    else:
        right_arm_msg = "Right Arm is Big"

    # Stomach check

    def abspross(LSx, LSy, LHx, LHy):
        midx = int(abs((LSx + LHx) / 2))
        midy = int(abs((LSy + LHy) / 2))
        chestx = int(abs((LSx + midx) / 2))
        chesty = int(abs((LSy + midy) / 2))
        absx = int(abs((midx + LHx) / 2))
        absy = int(abs((midy + LHy) / 2))

        chestdist = int(finddist(Simg, chestx, chesty, chestx, chesty))
        absdist = int(finddist(Simg, absx, absy, absx, absy))
        return (chestdist, absdist)

    chestdist, absdist = abspross(SLSx, SLSy, SLHx, SLHy)

    minchest = chestdist - 10
    maxchest = chestdist + 10

    if minchest <= absdist <= maxchest:
        abs_msg = "Abs are about the same as chest"

    elif absdist < minchest:
        abs_msg = "Abs and Chest have a Good Proportion "

    else:
        abs_msg = "Need to work on Abs too much fat "

    os.remove(pic1)
    os.remove(pic2)

    return res_shoulder_message, res_waist_message, right_arm_msg, left_arm_msg, abs_msg


img_path1 = "img/RF1.jpg"
img_path2 = "img/RS.jpg"
a, b, c, d, e = postpross(img_path1, img_path2)

print(a, b, c, d, e)

