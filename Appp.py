import streamlit as st
import cv2
import mediapipe as mp
import pickle
from collections import Counter
from streamlit_webrtc import webrtc_streamer, VideoTransformerBase

# ------------------- PAGE CONFIG -------------------
st.set_page_config(page_title="Gesture Recognition", layout="wide")

st.title("✋ Smart Hand Gesture Recognition System")
st.markdown("### Real-time Gesture Detection using Perceptron")

# ------------------- LOAD MODEL -------------------
model = pickle.load(open("model.pkl", "rb"))

# ------------------- MEDIAPIPE -------------------
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

mp_draw = mp.solutions.drawing_utils


# ------------------- VIDEO TRANSFORMER -------------------
class GestureTransformer(VideoTransformerBase):
    def __init__(self):
        self.predictions = []
        self.current_letter = ""

    def transform(self, frame):
        img = frame.to_ndarray(format="bgr24")
        img = cv2.flip(img, 1)

        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        result = hands.process(rgb)

        if result.multi_hand_landmarks:
            lm = result.multi_hand_landmarks[0].landmark

            base_x = lm[0].x
            base_y = lm[0].y

            landmarks = []
            for point in lm:
                landmarks.append(point.x - base_x)
                landmarks.append(point.y - base_y)

            # Prediction
            prediction = model.predict([landmarks])[0]
            self.predictions.append(prediction)

            if len(self.predictions) > 10:
                self.predictions.pop(0)

            # Smooth prediction
            self.current_letter = Counter(self.predictions).most_common(1)[0][0]

            # Draw landmarks
            for hand in result.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand, mp_hands.HAND_CONNECTIONS)

        # Display text
        cv2.putText(img, f"Gesture: {self.current_letter}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0), 2)

        return img


# ------------------- UI LAYOUT -------------------
col1, col2 = st.columns([2,1])

with col1:
    st.subheader("📷 Live Camera")
    webrtc_streamer(
        key="gesture",
        video_transformer_factory=GestureTransformer
    )

with col2:
    st.subheader("📊 System Info")
    st.success("✔ Model: Perceptron")
    st.info("👉 Show hand gesture clearly")
    st.warning("✋ Hold gesture steady for 1-2 sec")

    st.markdown("---")
    st.markdown("### 🧠 Features")
    st.write("• Real-time detection")
    st.write("• Smooth prediction")
    st.write("• MediaPipe hand tracking")
    st.write("• Streamlit WebRTC camera")
