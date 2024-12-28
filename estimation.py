import streamlit as st
from PIL import Image
import numpy as np
import cv2

DEMO_IMAGE = 'stand.jpg'

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

width = 368
height = 368
inWidth = width
inHeight = height

net = cv2.dnn.readNetFromTensorflow("graph_opt.pb")

# Sidebar for controls
st.sidebar.title("Controls")
img_file_buffer = st.sidebar.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])
thres = st.sidebar.slider('Threshold for key point detection', min_value=0, value=20, max_value=100, step=5) / 100

# Title and Instructions
st.title("Human Pose Estimation with OpenCV")
st.write("This app detects human poses in an image using OpenCV's pre-trained model.")
st.markdown("### Steps:")
st.markdown("1. Upload an image with all body parts clearly visible.")
st.markdown("2. Adjust the threshold for key point detection.")
st.markdown("3. View and download the result.")

# Load Image
if img_file_buffer is not None:
    image = np.array(Image.open(img_file_buffer))
else:
    image = np.array(Image.open(DEMO_IMAGE))

st.image(image, caption="Original Image", use_column_width=True)

@st.cache_data
def poseDetector(frame, threshold):
    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]

    net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]
        _, conf, _, point = cv2.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > threshold else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]
        if points[idFrom] and points[idTo]:
            cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv2.circle(frame, points[idFrom], 5, (0, 0, 255), cv2.FILLED)
            cv2.circle(frame, points[idTo], 5, (0, 0, 255), cv2.FILLED)

    return frame

# Perform Pose Detection
with st.spinner("Processing..."):
    output = poseDetector(image, thres)

st.image(output, caption="Pose Estimation", use_column_width=True)

# Download Button
result_image = Image.fromarray(cv2.cvtColor(output, cv2.COLOR_BGR2RGB))
st.download_button("Download Result", data=result_image.tobytes(), file_name="pose_estimation.png", mime="image/png")
