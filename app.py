import cv2
import numpy as np
import av
import streamlit as st
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from gesture_detector import is_middle_finger_up
from motion_tracker import MotionTracker
from keyword_listener import KeywordListener
import pygame
import time

# Initialize pygame mixer for audio
pygame.mixer.init()
middle_finger_sound = pygame.mixer.Sound("sounds/middle_finger.wav")
motion_sound = pygame.mixer.Sound("sounds/motion_67.wav")

# Track last play time for debouncing
last_middle_finger_time = 0
last_motion_time = 0

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

motion_tracker = MotionTracker()

# Helper function to play motion audio with cooldown
def play_motion_audio_if_ready():
    global last_motion_time
    current_time = time.time()
    if current_time - last_motion_time > 3.0:
        motion_sound.play()
        last_motion_time = current_time
        return True
    return False

def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    middle_finger_detected = False
    left_hand_y = None
    right_hand_y = None
    
    if results.multi_hand_landmarks and results.multi_handedness:
        for hand_landmarks, handedness in zip(results.multi_hand_landmarks, results.multi_handedness):
            mp_drawing.draw_landmarks(
                image,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing_styles.get_default_hand_landmarks_style(),
                mp_drawing_styles.get_default_hand_connections_style())
            
            # Check for middle finger gesture
            if is_middle_finger_up(hand_landmarks):
                middle_finger_detected = True
            
            # Track hand positions (using wrist landmark)
            hand_label = handedness.classification[0].label  # "Left" or "Right"
            wrist_y = hand_landmarks.landmark[0].y  # Wrist Y position
            
            if hand_label == "Left":
                left_hand_y = wrist_y
            else:
                right_hand_y = wrist_y
    
    # Update motion tracker
    motion_tracker.update(left_hand_y, right_hand_y)
    
    # Play audio for gestures (with 3 second cooldown)
    global last_middle_finger_time, last_motion_time
    current_time = time.time()
    
    # Display detection message before flipping so text is backwards (funny!)
    if middle_finger_detected:
        cv2.putText(image, "MIDDLE FINGER DETECTED! 🖕", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
        # Play audio if enough time has passed (3 second cooldown)
        if current_time - last_middle_finger_time > 3.0:
            middle_finger_sound.play()
            last_middle_finger_time = current_time
    
    # Display alternating motion detection
    if motion_tracker.alternating_detected:
        cv2.putText(image, "6 7! 👋👋", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
        # Play audio if enough time has passed (3 second cooldown)
        if current_time - last_motion_time > 3.0:
            motion_sound.play()
            last_motion_time = current_time
    
    return cv2.flip(image, 1)


RTC_CONFIGURATION = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

class VideoProcessor:
    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        img = process(img)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

webrtc_ctx = webrtc_streamer(
    key="WYH",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIGURATION,
    media_stream_constraints={"video": True, "audio": False},
    video_processor_factory=VideoProcessor,
    async_processing=True,
)

# Speech Recognition Section
st.markdown("---")
st.subheader("🎤 Live Keyword Detection + Voice Trigger")

# Initialize keyword listener and voice state
if 'keyword_listener' not in st.session_state:
    keywords = ["hello", "stop", "start", "yes", "no", "help", "six", "seven", "67"]
    st.session_state.keyword_listener = KeywordListener(keywords)
    st.session_state.keyword_listener.start()
    st.session_state.voice_six_detected = False
    st.session_state.voice_six_time = 0

# Display keywords
st.write("**Listening for keywords:**", ", ".join(st.session_state.keyword_listener.keywords))
st.info("🎯 **Voice Trigger**: Say 'six' then 'seven' OR just say '67' or 'six seven' to play audio!")

# Check for detected keywords
detected_keyword = st.session_state.keyword_listener.get_last_keyword()
current_time = time.time()

if detected_keyword:
    st.success(f"🎯 **Detected: {detected_keyword.upper()}**")
    
    # Handle "67" or combined "six seven" - immediate trigger
    if detected_keyword == "67":
        # Direct trigger - play audio immediately
        if play_motion_audio_if_ready():
            st.balloons()
            st.success("🎉 **AUDIO TRIGGERED BY VOICE!**")
    
    # Handle six-seven sequence
    elif detected_keyword == "six":
        st.session_state.voice_six_detected = True
        st.session_state.voice_six_time = current_time
        st.write("✅ 'Six' detected! Now say 'seven' within 3 seconds...")
    
    elif detected_keyword == "seven":
        # Check if "six" was said recently (within 3 seconds)
        if st.session_state.voice_six_detected and (current_time - st.session_state.voice_six_time < 3.0):
            # Trigger audio!
            if play_motion_audio_if_ready():
                st.balloons()
                st.success("🎉 **AUDIO TRIGGERED BY SIX+SEVEN!**")
            st.session_state.voice_six_detected = False
        else:
            st.write("⚠️ Say 'six' first, then 'seven' within 3 seconds")
else:
    # Reset if timeout
    if st.session_state.voice_six_detected and (current_time - st.session_state.voice_six_time > 3.0):
        st.session_state.voice_six_detected = False
        st.caption("⏱️ Voice combo timed out. Say 'six' again to start.")

# Auto-refresh to check for new keywords
time.sleep(0.1)
st.rerun()
