"""
Gesture detection module for hand gestures.
"""

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
