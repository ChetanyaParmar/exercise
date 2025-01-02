import cv2
import numpy as np
import time
import PoseModule as pm
import streamlit as st
from PIL import Image

# Set up Streamlit layout
st.title("Pose Detection App")
st.sidebar.title("Settings")
run_app = st.sidebar.checkbox("Run Pose Detection", value=True)

# Sidebar for additional settings (optional)
st.sidebar.markdown("Press **Stop Pose Detection** to stop the app.")

# Initialize counter and direction variables
count = 0
dir = 0

# Placeholder for Streamlit components
frame_placeholder = st.empty()
count_placeholder = st.sidebar.empty()
fps_placeholder = st.sidebar.empty()

# Initialize the pose detector
detector = pm.poseDetector()

# Store previous time for FPS calculation
pTime = 0

# Run the pose detection only when enabled in the sidebar
if run_app:
    # Initialize the video capture for webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Failed to open webcam.")
        st.stop()

    while run_app:
        success, img = cap.read()
        if not success:
            st.error("Error: Failed to read webcam frame.")
            break

        img = cv2.resize(img, (1280, 720))  # Resize image to a standard size
        img = detector.findPose(img, False)  # Find pose landmarks
        lmList = detector.findPosition(img, False)  # Get the list of landmarks

        if len(lmList) != 0:
            # Right Arm: Calculate the angle for the right arm
            angle = detector.findAngle(img, 12, 14, 16)

            # Interpolate percentage and bar position based on the angle
            per = np.interp(angle, (210, 310), (0, 100))
            bar = np.interp(angle, (220, 310), (650, 100))

            # Determine the color for drawing the progress bar
            color = (255, 0, 255)  # Default color for the progress bar
            if per == 100:
                color = (0, 255, 0)  # Green when the curl is complete
                if dir == 0:
                    count += 0.5
                    dir = 1
            if per == 0:
                color = (0, 255, 0)  # Green when returning to starting position
                if dir == 1:
                    count += 0.5
                    dir = 0

            # Draw progress bar
            cv2.rectangle(img, (1100, 100), (1175, 650), color, 3)
            cv2.rectangle(img, (1100, int(bar)), (1175, 650), color, cv2.FILLED)
            cv2.putText(img, f'{int(per)} %', (1100, 75), cv2.FONT_HERSHEY_PLAIN, 4, color, 4)

            # Draw curl count on the screen
            cv2.rectangle(img, (0, 450), (250, 720), (0, 255, 0), cv2.FILLED)
            cv2.putText(img, str(int(count)), (45, 670), cv2.FONT_HERSHEY_PLAIN, 15, (255, 0, 0), 25)

        # Calculate the FPS of the webcam feed
        cTime = time.time()
        fps = 1 / (cTime - pTime)
        pTime = cTime

        # Update the frame and counters in Streamlit
        fps_placeholder.markdown(f"**FPS**: {int(fps)}")
        count_placeholder.markdown(f"**Curl Count**: {int(count)}")

        # Convert BGR image to RGB for Streamlit
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        frame_placeholder.image(imgRGB, channels="RGB", use_column_width=True)

    # Release the video capture object
    cap.release()

st.sidebar.markdown("**App Status**: Stopped")
