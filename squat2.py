import cv2
import mediapipe as mp
import numpy as np
import streamlit as st

# Function to calculate the angle between three points
def findAngle(a, b, c, minVis=0.8):
    if a.visibility > minVis and b.visibility > minVis and c.visibility > minVis:
        bc = np.array([c.x - b.x, c.y - b.y, c.z - b.z])
        ba = np.array([a.x - b.x, a.y - b.y, a.z - b.z])

        angle = np.arccos((np.dot(ba, bc)) / (np.linalg.norm(ba) * np.linalg.norm(bc))) * (180 / np.pi)
        return angle
    return -1

# Function to determine the state of a leg
def legState(angle):
    if angle < 0:
        return 0  # Joint is not being picked up
    elif angle < 90:  # Adjust threshold for squats
        return 1  # Squat range
    elif angle < 140:
        return 2  # Transition range
    else:
        return 3  # Upright range

# Set up Streamlit layout
st.title("Squat Counter with Pose Detection")
st.sidebar.title("Settings")
run_app = st.sidebar.checkbox("Run Squat Counter", value=True)

# Sidebar elements for feedback
st.sidebar.markdown("Press **Stop Squat Counter** to stop the app.")
repCount = 0
lastState = 9

# Streamlit placeholders
frame_placeholder = st.empty()
count_placeholder = st.sidebar.empty()
fps_placeholder = st.sidebar.empty()

# Initialize MediaPipe Pose detector
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

if run_app:
    cap = cv2.VideoCapture(2)  # Access webcam

    if not cap.isOpened():
        st.error("Error: Could not open webcam.")
        st.stop()

    # Initialize pose detector
    with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while run_app:
            ret, frame = cap.read()
            if not ret or frame is None:
                st.error("Error: Failed to retrieve webcam frame.")
                break

            frame = cv2.resize(frame, (1280, 720))  # Resize frame
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)  # Pose detection

            if results and results.pose_landmarks:  # Ensure landmarks are detected
                lm_arr = results.pose_landmarks.landmark

                # Draw landmarks on the frame
                mp_drawing.draw_landmarks(
                    frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)
                )

                # Calculate angles for knees
                rAngle = findAngle(lm_arr[24], lm_arr[26], lm_arr[28])
                lAngle = findAngle(lm_arr[23], lm_arr[25], lm_arr[27])

                # Determine leg states
                rState = legState(rAngle)
                lState = legState(lAngle)
                state = rState * lState

                if state == 0:
                    st.warning("Legs not detected. Ensure you're in view of the camera.")
                elif state % 2 == 0 or rState != lState:
                    if lastState == 1:
                        st.info("Extend your legs fully.")
                    else:
                        st.info("Retract your legs.")
                else:
                    if state in (1, 9) and lastState != state:
                        lastState = state
                        if lastState == 1:
                            st.success("Good squat!")
                            repCount += 1

                count_placeholder.markdown(f"**Squats Completed**: {repCount}")

            else:
                st.warning("No landmarks detected. Make sure your whole body is visible.")

            # Display squat count on the video feed
            cv2.putText(
                frame, f"Squat Count: {repCount}", (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3, cv2.LINE_AA
            )

            # Display frame in Streamlit
            imgRGB = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_placeholder.image(imgRGB, channels="RGB", use_column_width=True)

        # Release video capture
        cap.release()

st.sidebar.markdown("**App Status**: Stopped")
