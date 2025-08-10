🖐 Hand Gesture Detection - Real-Time Recognition App
A Streamlit-powered web application for real-time hand gesture recognition using a pre-trained Convolutional Neural Network (CNN) model.
The app captures live video from your webcam, detects hands, processes frames, and predicts gestures with high accuracy — all in your browser.

✨ Features
📷 Live Webcam Input – Real-time video streaming from your webcam using streamlit-webrtc.
✋ Hand Detection – Contour-based detection to isolate the hand region and draw bounding boxes.
🤖 Gesture Prediction – Recognizes 10 unique hand gestures trained on the LeapGestRecog dataset.
⚙️ Configurable Options – Adjust prediction confidence threshold via a sidebar slider.
🛡 Prediction Smoothing – Reduces flicker with smoothing over multiple frames.
📈 FPS Monitoring – Displays frames-per-second for performance tracking.
🕘 Prediction History – Keeps track of recent predictions in real-time.

🛠 Tech Stack
Framework: Streamlit
Real-time Webcam: streamlit-webrtc
Model: TensorFlow/Keras CNN (trained on LeapGestRecog)
Image Processing: OpenCV for hand detection and drawing overlays
Video Processing: PyAV for handling webcam frames
Numerical Computing: NumPy

📂 Dataset
Source: LeapGestRecog on Kaggle

Classes (10 gestures):
01_palm, 02_l, 03_fist, 04_fist_moved, 05_thumb, 
06_index, 07_ok, 08_palm_moved, 09_c, 10_down
Image Size: 150×150 pixels (RGB)

Preprocessing: Normalized to [0, 1] range

Split: 80% training, 20% validation with data augmentation

🧠 Model Details
Architecture:

Input: (150, 150, 3)
Conv2D(32, (3,3), relu) → MaxPooling2D
Conv2D(64, (3,3), relu) → MaxPooling2D
Conv2D(128, (3,3), relu) → MaxPooling2D
Flatten → Dense(512, relu) → Dropout(0.5)
Dense(10, softmax)

Training:

Optimizer: Adam
Loss: Categorical Crossentropy
Metrics: Accuracy
Epochs: 30 (with EarlyStopping)
Augmentation: Rotation, shift, shear, zoom, flip

🚀 Installation & Setup
1️⃣ Clone this repository
git clone https://github.com/yourusername/gesture-recognition-app.git
cd gesture-recognition-app

2️⃣ Install dependencies
pip install streamlit streamlit-webrtc tensorflow opencv-python-headless numpy av

3️⃣ Place your trained model
gesture_model.h5
Put it in the root folder.

4️⃣ Run the app
streamlit run main.py

🎯 Usage
Open the app in your browser (default: http://localhost:8501)
Allow webcam access when prompted
Place your hand in front of the camera
See predictions and confidence scores in real time
Adjust the confidence threshold in the sidebar to filter uncertain predictions

⚠️ Limitations
Background Complexity – Contour-based detection may fail in noisy environments; consider integrating MediaPipe Hands for robustness.
Lighting – Low light can affect detection accuracy.
Performance – Real-time FPS may drop on low-end hardware.

🙏 Acknowledgments
Dataset: LeapGestRecog
Libraries: Streamlit, TensorFlow, OpenCV, PyAV, NumPy
