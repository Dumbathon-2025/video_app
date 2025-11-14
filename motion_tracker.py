"""
Motion tracking module for detecting alternating hand movements.
"""
from collections import deque


class MotionTracker:
    def __init__(self):
        self.left_hand_history = deque(maxlen=6)  # Short history for 6-7 pattern
        self.right_hand_history = deque(maxlen=6)
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
