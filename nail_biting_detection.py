import cv2
import mediapipe as mp
import numpy as np
import pyttsx3

# Initializing MediaPipe face and hand detection models
mp_face_detection = mp.solutions.face_detection
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

# Initializing the text-to-speech engine
engine = pyttsx3.init()

# Starting the video capture
cap = cv2.VideoCapture(0)

# Function to play the warning message
def play_warning():
    engine.say("Biting nails detected!")
    engine.runAndWait()

with mp_face_detection.FaceDetection(min_detection_confidence=0.2) as face_detection, \
     mp_hands.Hands(min_detection_confidence=0.2, min_tracking_confidence=0.2) as hands:

    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("Ignoring empty camera frame.")
            continue
        
        # Flipping the image horizontally for a later selfie-view display
        image = cv2.flip(image, 1)
        
        # Converting the BGR (Blue Green and Red) image to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Processing the image to find face
        face_results = face_detection.process(image_rgb)
        
        # Processing the image to find hands
        hand_results = hands.process(image_rgb)
        
        # face and hand annotations on the image
        if face_results.detections:
            for detection in face_results.detections:
                mp_drawing.draw_detection(image, detection)
                face_bbox = detection.location_data.relative_bounding_box
                face_cx = int((face_bbox.xmin + face_bbox.width / 2) * image.shape[1])
                face_cy = int((face_bbox.ymin + face_bbox.height / 2) * image.shape[0])

        if hand_results.multi_hand_landmarks:
            for hand_landmarks in hand_results.multi_hand_landmarks:
                mp_drawing.draw_landmarks(image, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                
                # fingertips to checks
                fingertip_landmarks = [
                    mp_hands.HandLandmark.THUMB_TIP,
                    mp_hands.HandLandmark.INDEX_FINGER_TIP,
                    mp_hands.HandLandmark.MIDDLE_FINGER_TIP,
                    mp_hands.HandLandmark.RING_FINGER_TIP,
                    mp_hands.HandLandmark.PINKY_TIP
                ]

                for fingertip in fingertip_landmarks:
                    finger_tip = hand_landmarks.landmark[fingertip]
                    finger_tip_x = int(finger_tip.x * image.shape[1])
                    finger_tip_y = int(finger_tip.y * image.shape[0])
                    
                    # Check if the finger tip is close to the face center
                    if face_results.detections:
                        distance = np.sqrt((finger_tip_x - face_cx) ** 2 + (finger_tip_y - face_cy) ** 2)
                        if distance < 50:  # adjust this threshold
                            # print("Warning: Biting nails detected!")
                           # cv2.putText(image, "Warning: Biting nails detected!", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2, cv2.LINE_AA)
                            play_warning()
                            break  
        
       
        cv2.imshow('Nail Biting Detection', image)
        
        if cv2.waitKey(5) & 0xFF == 27:
            break
        
        if cv2.getWindowProperty('Nail Biting Detection', cv2.WND_PROP_VISIBLE) < 1:
                break

    cap.release()
    cv2.destroyAllWindows()
cap.release()
cv2.destroyAllWindows()
