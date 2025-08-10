import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
import cv2
import numpy as np
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av
import time
from collections import deque

# Load the trained model with error handling
@st.cache_resource
def load_gesture_model():
    try:
        model = load_model('gesture_model.h5')
        st.success("Model loaded successfully!")
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

# Preprocess the frame for prediction
def preprocess_frame(frame, roi=None):
    # If ROI (region of interest) is provided, crop to it
    if roi is not None:
        x, y, w, h = roi
        frame = frame[y:y+h, x:x+w]
        if frame.size == 0:  # Handle empty ROI
            return None
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # Resize to 150x150
    frame_resized = cv2.resize(frame_rgb, (150, 150))
    # Convert to array and normalize
    frame_array = img_to_array(frame_resized)
    frame_array = frame_array / 255.0
    # Expand dimensions to match model input shape (1, 150, 150, 3)
    frame_array = np.expand_dims(frame_array, axis=0)
    return frame_array

# Detect hand region using contour-based approach
def detect_hand(frame):
    # Convert to grayscale and apply Gaussian blur
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Threshold the image
    _, thresh = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if contours:
        # Get the largest contour (assumed to be the hand)
        largest_contour = max(contours, key=cv2.contourArea)
        if cv2.contourArea(largest_contour) > 500:  # Minimum area to avoid noise
            x, y, w, h = cv2.boundingRect(largest_contour)
            return (x, y, w, h)
    return None

# Define class labels (modify based on your dataset folder names)
class_labels = [
    '01_palm', '02_l', '03_fist', '04_fist_moved', '05_thumb',
    '06_index', '07_ok', '08_palm_moved', '09_c', '10_down'
]

# Video processor for webcam feed
class GestureProcessor(VideoProcessorBase):
    def __init__(self):
        self.model = load_gesture_model()
        self.prediction = "No prediction"
        self.confidence = 0.0
        self.prediction_queue = deque(maxlen=5)  # Smooth over 5 frames
        self.confidence_threshold = 0.7  # Default threshold
        self.fps = 0
        self.frame_count = 0
        self.start_time = time.time()

    def update_fps(self):
        self.frame_count += 1
        elapsed = time.time() - self.start_time
        if elapsed > 1.0:
            self.fps = self.frame_count / elapsed
            self.frame_count = 0
            self.start_time = time.time()

    def recv(self, frame):
        if self.model is None:
            return frame

        # Convert frame to numpy array
        img = frame.to_ndarray(format="bgr24")
        
        # Detect hand region
        roi = detect_hand(img)
        if roi is not None:
            x, y, w, h = roi
            # Draw bounding box
            cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)
            # Preprocess and predict
            processed_frame = preprocess_frame(img, roi)
            if processed_frame is not None:
                prediction = self.model.predict(processed_frame, verbose=0)[0]
                max_confidence = np.max(prediction)
                if max_confidence >= self.confidence_threshold:
                    predicted_class = class_labels[np.argmax(prediction)]
                    self.prediction_queue.append((predicted_class, max_confidence))
        
        # Smooth predictions
        if self.prediction_queue:
            # Get most common prediction in queue
            predictions = [p[0] for p in self.prediction_queue]
            confidences = [p[1] for p in self.prediction_queue]
            self.prediction = max(set(predictions), key=predictions.count)
            self.confidence = np.mean(confidences) * 100

        # Update FPS
        self.update_fps()

        # Draw prediction and FPS on frame
        cv2.putText(img, f"Gesture: {self.prediction} ({self.confidence:.2f}%)",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        cv2.putText(img, f"FPS: {self.fps:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# Streamlit app
st.title("Live Gesture Recognition App")
st.write("Use your webcam to capture hand gestures and predict them in real-time using the trained CNN model.")

# Sidebar for configuration
st.sidebar.header("Configuration")
confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.7, 0.05)
st.sidebar.write("Predictions with confidence below this threshold will be ignored.")

# WebRTC configuration
RTC_CONFIGURATION = RTCConfiguration({"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]})

# Start webcam stream
ctx = webrtc_streamer(
    key="gesture-recognition",
    video_processor_factory=GestureProcessor,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False}
)

# Update confidence threshold dynamically
if ctx.video_processor:
    ctx.video_processor.confidence_threshold = confidence_threshold

# Display prediction and history
if ctx.video_processor:
    st.subheader("Current Prediction")
    st.write(f"**Gesture:** {ctx.video_processor.prediction}")
    st.write(f"**Confidence:** {ctx.video_processor.confidence:.2f}%")
    st.write(f"**FPS:** {ctx.video_processor.fps:.2f}")

    st.subheader("Prediction History")
    if ctx.video_processor.prediction_queue:
        history = list(ctx.video_processor.prediction_queue)
        for i, (pred, conf) in enumerate(reversed(history)):
            st.write(f"Frame {i+1}: {pred} ({conf*100:.2f}%)")

# Instructions
st.write("""
### Instructions
1. Allow webcam access when prompted.
2. Position your hand clearly in front of the webcam.
3. A green bounding box will highlight the detected hand region.
4. Predictions above the confidence threshold (adjustable in the sidebar) are displayed.
5. The app smooths predictions over 5 frames to reduce flicker.
6. FPS is shown to monitor performance.
""")