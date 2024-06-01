import os
import cv2
import mediapipe as mp
import numpy as np
import pyautogui
import time
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
pyautogui.FAILSAFE = False

mp_hands = mp.solutions.hands
hands = mp_hands.Hands()
mp_drawing = mp.solutions.drawing_utils

devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

min_volume, max_volume, _ = volume.GetVolumeRange()

def euclidean_distance(point1, point2):
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

def detect_gesture(landmarks):
    thumb_tip = landmarks[4]
    index_tip = landmarks[8]
    middle_tip = landmarks[12]
    ring_tip = landmarks[16]
    pinky_tip = landmarks[20]

    # Gesture 1: Thumb and Index finger touching (Mouse Click)
    if euclidean_distance(thumb_tip, index_tip) < 0.03:
        return "Mouse Click"

    # Gesture 2: Thumb touching Ring finger (Volume Mute)
    if euclidean_distance(thumb_tip, ring_tip) < 0.05:
        return "Volume Down"

    # Gesture 3: Thumb touching Middle finger (Volume Up)
    if euclidean_distance(thumb_tip, middle_tip) < 0.05:
        return "Volume Up"

    # Gesture 4: Thumb touching Pinky finger (Volume Down)
    if euclidean_distance(thumb_tip, pinky_tip) < 0.05:
        return "Volume Mute"

    # Gesture 5: All fingers extended and spread apart (Volume Unmute)
    if (euclidean_distance(thumb_tip, index_tip) > 0.1 and
        euclidean_distance(index_tip, middle_tip) > 0.1 and
        euclidean_distance(middle_tip, ring_tip) > 0.1 and
        euclidean_distance(ring_tip, pinky_tip) > 0.1):
        return "Volume Unmute"

    return "Unknown"

last_click_time = 0
double_click_threshold = 0.5

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Error: Failed to capture image.")
        break
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    result = hands.process(image_rgb)
    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            landmarks = [(lm.x, lm.y) for lm in hand_landmarks.landmark]
            gesture = detect_gesture(landmarks)
            current_time = time.time()
            if gesture == "Mouse Click":
                if current_time - last_click_time < double_click_threshold:
                    pyautogui.doubleClick()
                    print("Performing Mouse Double Click")
                else:
                    pyautogui.click()
                    print("Performing Mouse Click")
                last_click_time = current_time
            elif gesture == "Volume Mute":
                volume.SetMute(1, None)
                print("Muting Volume")
            elif gesture == "Volume Unmute":
                volume.SetMute(0, None)
                print("Unmuting Volume")
            elif gesture == "Volume Up":
                current_volume = volume.GetMasterVolumeLevel()
                new_volume = min(max_volume, current_volume + 2.0)  # Ensure volume doesn't exceed max
                volume.SetMasterVolumeLevel(new_volume, None)
                print("Increasing Volume")
            elif gesture == "Volume Down":
                current_volume = volume.GetMasterVolumeLevel()
                new_volume = max(min_volume, current_volume - 2.0)  # Ensure volume doesn't go below min
                volume.SetMasterVolumeLevel(new_volume, None)
                print("Decreasing Volume")
            else:
                print("Gesture Unknown")

    #cv2.imshow('MediaPipe Hands', image)
    if cv2.waitKey(5) & 0xFF == 27:
        break

cap.release()
cv2.destroyAllWindows()
