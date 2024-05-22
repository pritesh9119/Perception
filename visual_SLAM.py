import sys
import cv2
import numpy as np
import orbslam2

# Path to the ORB-SLAM2 vocabulary file
VOCABULARY_PATH = "ORBvoc.txt"
# Path to the ORB-SLAM2 configuration file
CONFIG_FILE_PATH = "path_to_config.yaml"

def main():
    # Initialize the ORB-SLAM2 system
    slam = orbslam2.System(VOCABULARY_PATH, CONFIG_FILE_PATH, orbslam2.Sensor.MONOCULAR)
    slam.initialize()

    # Open a video capture object
    cap = cv2.VideoCapture(0)  # Change 0 to the path of a video file if needed

    if not cap.isOpened():
        print("Error: Could not open video.")
        return

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        # Convert the frame to grayscale as ORB-SLAM2 works with grayscale images
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Pass the frame to the SLAM system
        slam.process_image_mono(gray_frame, cv2.getTickCount() / cv2.getTickFrequency())

        # Get the current pose of the camera
        pose = slam.get_camera_pose()
        print("Current Camera Pose:", pose)

        # Display the current frame
        cv2.imshow('Frame', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # Shutdown the SLAM system
    slam.shutdown()

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == '__main__':
    main()
