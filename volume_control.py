import cv2
import numpy as np
import mediapipe as mp
from math import hypot
from ctypes import cast, POINTER
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

# ================== AUDIO SETUP ==================
devices = AudioUtilities.GetSpeakers()
interface = devices.Activate(
    IAudioEndpointVolume._iid_, CLSCTX_ALL, None)
volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_range = volume.GetVolumeRange()
min_vol = vol_range[0]
max_vol = vol_range[1]

# ================== MEDIAPIPE SETUP ==================
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(min_detection_confidence=0.7,
                       min_tracking_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# ================== CAMERA ==================
cap = cv2.VideoCapture(0)
cap.set(3, 640)
cap.set(4, 480)

print("Press 'Q' to quit.")

while True:
    success, img = cap.read()
    if not success:
        break

    img = cv2.flip(img, 1)
    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb_img)

    lm_list = []

    if result.multi_hand_landmarks:
        for hand_landmarks in result.multi_hand_landmarks:
            for id, lm in enumerate(hand_landmarks.landmark):
                h, w, c = img.shape
                cx, cy = int(lm.x * w), int(lm.y * h)
                lm_list.append([id, cx, cy])

            # Drawing hand landmarks
            mp_draw.draw_landmarks(
                img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    if lm_list != []:
        # Thumb tip (4) & Index finger tip (8)
        x1, y1 = lm_list[4][1], lm_list[4][2]
        x2, y2 = lm_list[8][1], lm_list[8][2]

        # Draw points
        cv2.circle(img, (x1, y1), 10, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 10, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 255), 3)

        # Distance between fingers
        length = hypot(x2 - x1, y2 - y1)

        # Volume range
        vol = np.interp(length, [20, 200], [min_vol, max_vol])
        vol_bar = np.interp(length, [20, 200], [400, 150])
        vol_percent = np.interp(length, [20, 200], [0, 100])

        # Set Volume
        volume.SetMasterVolumeLevel(vol, None)

        # Volume Bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 255, 0), 3)
        cv2.rectangle(img, (50, int(vol_bar)),
                      (85, 400), (0, 255, 0), cv2.FILLED)

        # Volume Percentage
        cv2.putText(img, f'{int(vol_percent)} %',
                    (40, 430), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 3)

        # Extra visual feedback
        if length < 40:
            cv2.circle(img, ((x1 + x2)//2,
                       (y1 + y2)//2), 10, (0, 255, 0), cv2.FILLED)

    # Display
    cv2.imshow("HAND GESTURE VOLUME CONTROL", img)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
