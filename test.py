import cv2

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Cannot open camera")
    exit()

while True:
    success, img = cap.read()

    if not success:
        print("Failed to read from camera")
        break

    cv2.imshow("Camera Test", img)

    key = cv2.waitKey(1)

    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()