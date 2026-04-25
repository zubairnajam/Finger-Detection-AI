"""
Webcam Test Script
Tests if OpenCV can access your webcam and display video feed
Press 'q' to quit
"""

import cv2

def test_webcam():
    """Test webcam access and display"""
    print("Testing webcam access...")
    print("Press 'q' to quit the video feed")
    
    # Initialize webcam (0 is usually the default camera)
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        print("❌ Error: Could not open webcam")
        print("Troubleshooting:")
        print("  - Check if your webcam is connected")
        print("  - Check if another application is using the webcam")
        print("  - Try changing the camera index (0, 1, 2, etc.)")
        return False
    
    print("✓ Webcam opened successfully")
    print("✓ Displaying video feed... (press 'q' to quit)")
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        
        if not ret:
            print("❌ Error: Could not read frame")
            break
        
        frame_count += 1
        
        # Add some info overlay
        cv2.putText(frame, f"Frame: {frame_count}", (10, 30), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.putText(frame, "Press 'q' to quit", (10, 60), 
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Display the frame
        cv2.imshow('Webcam Test', frame)
        
        # Exit on 'q' key press
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Cleanup
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n✅ Test completed! Processed {frame_count} frames")
    return True

if __name__ == "__main__":
    test_webcam()