"""
MediaPipe Hand Detection Test (Updated for MediaPipe 0.10.9+)
Tests MediaPipe Hands library with live webcam feed
Shows hand landmarks and finger detection
Press 'q' to quit
"""

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

def draw_landmarks_on_image(rgb_image, detection_result):
    """Draw hand landmarks on the image"""
    hand_landmarks_list = detection_result.hand_landmarks
    annotated_image = np.copy(rgb_image)

    # Loop through the detected hands to visualize.
    for hand_landmarks in hand_landmarks_list:
        vision.drawing_utils.draw_landmarks(
            annotated_image,
            hand_landmarks,
            vision.HandLandmarksConnections.HAND_CONNECTIONS,
            vision.drawing_styles.get_default_hand_landmarks_style(),
            vision.drawing_styles.get_default_hand_connections_style(),
        )

    return annotated_image

def test_mediapipe():
    """Test MediaPipe hand detection on webcam feed"""
    print("Testing MediaPipe Hands...")
    print("Press 'q' to quit")
    
    # Download the model file if needed
    # You can download from: https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task
    model_path = 'hand_landmarker.task'
    
    try:
        # Create HandLandmarker
        base_options = python.BaseOptions(model_asset_path=model_path)
        options = vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=vision.RunningMode.VIDEO,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        detector = vision.HandLandmarker.create_from_options(options)
        
    except Exception as e:
        print(f"Error initializing MediaPipe: {e}")
        print("\nThe model file 'hand_landmarker.task' is missing.")
        print("Please download it from:")
        print("https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task")
        print("\nSave it in the same directory as this script.")
        return False
    
    # Initialize webcam
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("Error: Could not open webcam")
        return False
    
    print("Webcam opened successfully")
    print("MediaPipe Hands initialized")
    print("Show your hand to the camera...")
    
    frame_count = 0
    hands_detected = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Flip frame horizontally for mirror view
        frame = cv2.flip(frame, 1)
        
        # Convert BGR to RGB for MediaPipe
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Create MediaPipe Image
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)
        
        # Process the frame with timestamp
        timestamp_ms = frame_count * 33
        detection_result = detector.detect_for_video(mp_image, timestamp_ms)
        
        # Draw landmarks if hand detected
        if detection_result.hand_landmarks:
            hands_detected += 1
            annotated_image = draw_landmarks_on_image(rgb_frame, detection_result)
            frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)
            status_text = "Hand Detected!"
            status_color = (0, 255, 0)  # Green
        else:
            status_text = "No hand detected"
            status_color = (0, 0, 255)  # Red
        
        # Add overlay text
        cv2.putText(frame, status_text, (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, status_color, 2)
        cv2.putText(frame, f"Frames: {frame_count}", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        cv2.putText(frame, "Press 'q' to quit", (10, 90), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display the frame
        cv2.imshow('MediaPipe Hand Detection Test', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    detector.close()
    
    print("\nTest completed!")
    print(f"   Total frames: {frame_count}")
    print(f"   Hands detected: {hands_detected} frames")
    if frame_count > 0:
        print(f"   Detection rate: {(hands_detected/frame_count)*100:.1f}%")
    
    return True

if __name__ == "__main__":
    test_mediapipe()
