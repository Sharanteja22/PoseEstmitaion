import mediapipe as mp  
import cv2

# Import the new drawing utilities
from mediapipe.python.solutions.drawing_utils import DrawingSpec, draw_landmarks

# Initialize MediaPipe pose
mp_pose = mp.solutions.pose
pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)

# Load the image
image_path = "C:\\Users\\tejac\\OneDrive\\Desktop\\AICTEE\\week2\\images.png"
image = cv2.imread(image_path)
image_rgv = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
results = pose.process(image_rgv)

if results.pose_landmarks:
    print("Pose landmarks detected!")

    # Print pose landmarks
    for idx, landmark in enumerate(results.pose_landmarks.landmark):
        print(f"Landmark {idx}: (x: {landmark.x}, y: {landmark.y}, z: {landmark.z})")

    # Draw landmarks on the image
    h, w, c = image.shape
    for landmark in results.pose_landmarks.landmark:
        cx, cy = int(landmark.x * w), int(landmark.y * h)
        cv2.circle(image, (cx, cy), 5, (255, 0, 0), -1)

    # Annotate the image using the new `draw_landmarks` method
    annotated_image = image.copy()
    draw_landmarks(
        annotated_image,
        results.pose_landmarks,
        mp_pose.POSE_CONNECTIONS,
        DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),  # Styling
        DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=2)   # Styling for connections
    )

    # Display the annotated image
cv2.imshow("Pose", annotated_image)
cv2.imshow("PoseLandmark",image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# Close the pose detector
pose.close()
