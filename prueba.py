import cv2
import mediapipe as mp
import numpy as np

# Initialize Mediapipe FaceMesh
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh()

# Load the video
cap = cv2.VideoCapture("P1M3_edited.mp4")

# Flag to check if the video has been played
video_played = False

while True:
    ret, frame = cap.read()
    if not ret:
        # If the video has reached the end, break out of the loop
        break

    # Convert the frame to RGB for Mediapipe
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process the frame with Mediapipe FaceMesh
    results = face_mesh.process(rgb_frame)

    if results.multi_face_landmarks:
        for face_landmarks in results.multi_face_landmarks:
            # Extract forehead landmarks
            forehead_landmarks = [
                face_landmarks.landmark[10],  # Left forehead
                face_landmarks.landmark[105],
                face_landmarks.landmark[334]# Right forehead
            ]

            # Convert forehead landmarks to pixel coordinates
            h, w, _ = frame.shape
            forehead_points = np.array([(int(l.x * w), int(l.y * h)) for l in forehead_landmarks])
            
            
            margin = 4
            
            # Define a custom forehead region
            forehead_top = forehead_points[1][1] - 4*margin # Top of forehead
            forehead_left = min(forehead_points[0][0], forehead_points[1][0]) # Leftmost point
            forehead_right = max(forehead_points[2][0], forehead_points[1][0]) # Rightmost point
            forehead_bottom = min(forehead_points[0][1], forehead_points[2][1]) - margin# Bottom of forehead
            
            # Draw rectangle around forehead region on the original image
            cv2.rectangle(frame, (forehead_left, forehead_top), (forehead_right, forehead_bottom), (0, 255, 0), 2)

    
    # Display the frame with forehead recognition
    cv2.imshow("Forehead Recognition", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
    # Check if the window is still open before showing a new frame
    if cv2.getWindowProperty("Forehead Recognition", cv2.WND_PROP_VISIBLE) < 1:
        break
    
# Release the video capture object
cap.release()

# Wait for a key press before closing the window
cv2.waitKey(0)

# Close all OpenCV windows
cv2.destroyAllWindows()
