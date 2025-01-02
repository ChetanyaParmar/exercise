import cv2
import numpy as np
import time
import mediapipe as mp
import streamlit as st
from PIL import Image

# Set up Streamlit layout
st.title("Push-up Counter App with Pose Detection")
st.sidebar.title("Settings")
run_app = st.sidebar.checkbox("Run Push-up Counter", value=True)

# Sidebar for additional settings (optional)
st.sidebar.markdown("Press **Stop Push-up Counter** to stop the app.")

# Initialize push-up count and direction variables
pushUpStart = 0
pushUpCount = 0

# Placeholder for Streamlit components
frame_placeholder = st.empty()
count_placeholder = st.sidebar.empty()
fps_placeholder = st.sidebar.empty()

# Initialize Mediapipe Pose detector
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Function to calculate distance between two points (used for push-up detection)
def distanceCalculate(p1, p2):
    dis = ((p2[0] - p1[0]) ** 2 + (p2[1] - p1[1]) ** 2) ** 0.5
    return dis

# Store previous time for FPS calculation
pTime = 0

# Run the push-up counter only when enabled in the sidebar
if run_app:
    # Initialize the video capture for webcam
    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        st.error("Error: Failed to open webcam.")
        st.stop()

    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while run_app:
            success, image = cap.read()
            if not success:
                st.error("Error: Failed to read webcam frame.")
                break

            image = cv2.resize(image, (1280, 720))  # Resize image to a standard size
            image.flags.writeable = False  # Mark the image as read-only for Mediapipe processing
            results = pose.process(image)
            image.flags.writeable = True  # Mark the image as writable after Mediapipe processing
            image_height, image_width, _ = image.shape

            if results.pose_landmarks:
                # Extract relevant landmarks for pose detection
                rightShoulder = (int(results.pose_landmarks.landmark[12].x * image_width),
                                 int(results.pose_landmarks.landmark[12].y * image_height))
                rightWrist = (int(results.pose_landmarks.landmark[16].x * image_width),
                              int(results.pose_landmarks.landmark[16].y * image_height))

                # Push-up detection logic
                if distanceCalculate(rightShoulder, rightWrist) < 130:
                    pushUpStart = 1
                elif pushUpStart and distanceCalculate(rightShoulder, rightWrist) > 250:
                    pushUpCount += 1
                    pushUpStart = 0

                # Draw push-up count on the image
                font = cv2.FONT_HERSHEY_SIMPLEX
                org = (100, 100)
                fontScale = 1
                color = (255, 0, 0)  # Blue color in BGR
                thickness = 2
                image = cv2.putText(image, "Push-up count:  " + str(pushUpCount), org, font, fontScale, color, thickness,
                                    cv2.LINE_AA)

                # Optionally, draw pose landmarks on the image
                mp_drawing.draw_landmarks(image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)

            # Calculate the FPS of the webcam feed
            cTime = time.time()
            fps = 1 / (cTime - pTime)
            pTime = cTime

            # Update the frame and counters in Streamlit
            fps_placeholder.markdown(f"**FPS**: {int(fps)}")
            count_placeholder.markdown(f"**Push-up Count**: {int(pushUpCount)}")

            # Convert BGR image to RGB for Streamlit display
            imgRGB = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(imgRGB, channels="RGB", use_column_width=True)

        # Release the video capture object
        cap.release()

st.sidebar.markdown("**App Status**: Stopped")
