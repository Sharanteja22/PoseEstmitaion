import mediapipe as mp
import cv2

# Import the new drawing utilities
from mediapipe.python.solutions.drawing_utils import DrawingSpec, draw_landmarks

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=False, min_detection_confidence=0.5, min_tracking_confidence=0.5)

# Start video capture from the webcam
cap = cv2.VideoCapture(0)  # 0 is the default channel for the primary webcam
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 5000)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 2500)
while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture video")
        break

    # Convert the frame to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(frame_rgb)

    if results.pose_landmarks:
        # Draw pose landmarks on the frame
        draw_landmarks(
            frame,
            results.pose_landmarks,
            mp_pose.POSE_CONNECTIONS,
            DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Styling
            DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Styling for connections
        )

    # Display the video feed with landmarks
    cv2.imshow("Pose Detection", frame)

    # Exit loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture and close windows
cap.release()
cv2.destroyAllWindows()
pose.close()
