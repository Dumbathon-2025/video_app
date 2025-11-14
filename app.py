import cv2
import numpy as np
import av
import mediapipe as mp
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
from collections import deque

mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    model_complexity=0,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5,
    max_num_hands=2
)

# Motion tracking state
class MotionTracker:
    def __init__(self):
        self.left_hand_history = deque(maxlen=8)  # Very short history
        self.right_hand_history = deque(maxlen=8)
        self.alternating_detected = False
        self.detection_frames = 0  # Count frames where alternating is detected
        self.cooldown_frames = 0  # Frames to keep showing after detection
        
    def update(self, left_y, right_y):
        """Update hand positions and detect alternating motion"""
        # Both hands must be present
        if left_y is None or right_y is None:
            self.left_hand_history.clear()
            self.right_hand_history.clear()
            # Keep showing if we have cooldown frames left
            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1
                self.alternating_detected = True
            else:
                self.alternating_detected = False
                self.detection_frames = 0
            return
            
        self.left_hand_history.append(left_y)
        self.right_hand_history.append(right_y)
            
        # Need at least 4 frames
        if len(self.left_hand_history) < 4 or len(self.right_hand_history) < 4:
            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1
                self.alternating_detected = True
            else:
                self.alternating_detected = False
            return
        
        # Compare first half vs second half to detect direction
        left_list = list(self.left_hand_history)
        right_list = list(self.right_hand_history)
        
        # Recent movement (last 4 frames)
        left_movement = left_list[-1] - left_list[-4]
        right_movement = right_list[-1] - right_list[-4]
        
        movement_threshold = 0.008  # Lowered threshold for more sensitivity
        
        # Check if both hands are moving with enough motion
        left_is_moving = abs(left_movement) > movement_threshold
        right_is_moving = abs(right_movement) > movement_threshold
        
        # Check if they're moving in opposite directions
        if left_is_moving and right_is_moving and (left_movement * right_movement) < 0:
            self.detection_frames += 1
            self.cooldown_frames = 30  # Show for 30 frames (~1 sec) after last detection
        else:
            if self.detection_frames > 0:
                self.detection_frames -= 0.5  # Decay slowly
        
        # Trigger if we've detected it for at least 1.5 frames OR we're in cooldown
        if self.detection_frames >= 1.5 or self.cooldown_frames > 0:
            self.alternating_detected = True
            if self.cooldown_frames > 0:
                self.cooldown_frames -= 1
        else:
            self.alternating_detected = False

motion_tracker = MotionTracker()

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
