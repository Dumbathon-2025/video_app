import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def is_middle_finger_up(hand_landmarks):
    """
    Detect if middle finger is extended while other fingers are folded.
    Returns True if middle finger gesture is detected.
    """
    # Get landmark positions
    landmarks = hand_landmarks.landmark
    
    # Finger tip and base landmarks
    # Thumb: 4 (tip), 3, 2
    # Index: 8 (tip), 6 (base)
    # Middle: 12 (tip), 10 (base)
    # Ring: 16 (tip), 14 (base)
    # Pinky: 20 (tip), 18 (base)
    
    # Check if middle finger is extended (tip higher than base)
    middle_extended = landmarks[12].y < landmarks[10].y
    
    # Check if other fingers are folded (tips not higher than their middle joints)
    index_folded = landmarks[8].y > landmarks[6].y
    ring_folded = landmarks[16].y > landmarks[14].y
    pinky_folded = landmarks[20].y > landmarks[18].y
    thumb_folded = landmarks[4].y > landmarks[3].y or abs(landmarks[4].x - landmarks[3].x) < 0.05
    
    # Middle finger up, others down
    return middle_extended and index_folded and ring_folded and pinky_folded

def process(image):
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = hands.process(image)

    # Draw the hand annotations on the image.
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    middle_finger_detected = False
    
    if results.multi_hand_landmarks:
      for hand_landmarks in results.multi_hand_landmarks:
        mp_drawing.draw_landmarks(
            image,
            hand_landmarks,
            mp_hands.HAND_CONNECTIONS,
            mp_drawing_styles.get_default_hand_landmarks_style(),
            mp_drawing_styles.get_default_hand_connections_style())
        
        # Check for middle finger gesture
        if is_middle_finger_up(hand_landmarks):
            middle_finger_detected = True
    
    # Display detection message before flipping so text is backwards (funny!)
    if middle_finger_detected:
        cv2.putText(image, "MIDDLE FINGER DETECTED! 🖕", (10, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 3)
    
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
