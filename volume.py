import cv2
import numpy as np
import mediapipe as mp
from math import hypot
from ctypes import POINTER, cast
from comtypes import CLSCTX_ALL
from pycaw.pycaw import AudioUtilities, IAudioEndpointVolume

print("Starting...")

# ---------------- VOLUME CONTROL ---------------- #
devices = AudioUtilities.GetSpeakers()

interface = devices._dev.Activate(
    IAudioEndpointVolume._iid_,
    CLSCTX_ALL,
    None
)

volume = cast(interface, POINTER(IAudioEndpointVolume))

vol_min, vol_max = volume.GetVolumeRange()[:2]

# ---------------- MEDIAPIPE ---------------- #
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

mp_draw = mp.solutions.drawing_utils

# ---------------- CAMERA ---------------- #
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Camera not opening")
    exit()

print("Camera opened successfully")

while True:
    success, img = cap.read()

    if not success:
        print("Failed to read camera")
        break

    img = cv2.flip(img, 1)

    rgb_img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb_img)

    lm_list = []

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:

            mp_draw.draw_landmarks(
                img,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS
            )

            h, w, c = img.shape

            for idx, landmark in enumerate(hand_landmarks.landmark):
                cx = int(landmark.x * w)
                cy = int(landmark.y * h)
                lm_list.append((idx, cx, cy))

    if len(lm_list) != 0:
        # Thumb tip
        x1, y1 = lm_list[4][1], lm_list[4][2]

        # Index finger tip
        x2, y2 = lm_list[8][1], lm_list[8][2]

        # Draw circles and line
        cv2.circle(img, (x1, y1), 12, (255, 0, 255), cv2.FILLED)
        cv2.circle(img, (x2, y2), 12, (255, 0, 255), cv2.FILLED)
        cv2.line(img, (x1, y1), (x2, y2), (0, 255, 0), 3)

        # Find distance
        length = hypot(x2 - x1, y2 - y1)

        # Convert hand distance to system volume
        system_volume = np.interp(length, [20, 250], [vol_min, vol_max])
        volume.SetMasterVolumeLevel(system_volume, None)

        # Convert distance to percentage
        volume_percent = np.interp(length, [20, 250], [0, 100])

        # Volume bar position
        vol_bar = np.interp(length, [20, 250], [400, 150])

        # Draw volume bar
        cv2.rectangle(img, (50, 150), (85, 400), (0, 0, 255), 3)
        cv2.rectangle(img, (50, int(vol_bar)), (85, 400), (0, 0, 255), cv2.FILLED)

        # Display percentage
        cv2.putText(
            img,
            f'{int(volume_percent)}%',
            (35, 450),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            3
        )

    # Show title
    cv2.putText(
        img,
        "Hand Gesture Volume Control",
        (300, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        1,
        (0, 255, 255),
        3
    )

    cv2.imshow("Hand Gesture Volume Control", img)

    # Exit on Q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()