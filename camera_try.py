import cv2
import time
import os
# Open the webcam (0 is usually the default webcam, you can change it if needed)
device_id = 7  # Update if needed

# Apply manual settings
os.system(f"v4l2-ctl -d /dev/video{device_id} --set-ctrl=auto_exposure=1")
os.system(f"v4l2-ctl -d /dev/video{device_id} --set-ctrl=exposure_time_absolute=1000")
os.system(f"v4l2-ctl -d /dev/video{device_id} --set-ctrl=gain=50")
os.system(f"v4l2-ctl -d /dev/video{device_id} --set-ctrl=white_balance_automatic=0")
os.system(f"v4l2-ctl -d /dev/video{device_id} --set-ctrl=white_balance_temperature=4500")

cap = cv2.VideoCapture(device_id, cv2.CAP_V4L2)

# Check if the camera opened successfully
if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

t0 = time.time()
i=0
print(t0)
for _ in range(5):  # Adjust number of iterations if needed

    ret, frame = cap.read()
while i < 5:
    

    if time.time() - t0 >= i+1:
        print(time.time())
        for _ in range(5):  # Adjust number of iterations if needed
            ret, frame = cap.read()
        print("Capturing image...")
        # Save the captured image
        print(frame.shape)
        cv2.imwrite(f"captured_image_{i+1}.jpg", frame)
        print("Image saved")
        i += 1

# Release the webcam and close any OpenCV windows
cap.release()
cv2.destroyAllWindows()
