import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from gesture_detector import is_middle_finger_up
from motion_tracker import MotionTracker

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
    
    # Display detection message before flipping so text is backwards (funny!)
    if middle_finger_detected:
        cv2.putText(image, "MIDDLE FINGER DETECTED! 🖕", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
    # Display alternating motion detection
    if motion_tracker.alternating_detected:
        cv2.putText(image, "ALTERNATING MOTION! 👋👋", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 3)
    
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
