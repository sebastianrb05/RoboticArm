import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
import mediapipe as mp
from pyfirmata2 import Arduino, SERVO



# Define Arduino
board = Arduino('COM3')

# Define Servo
thumbPin = board.digital[9]
pointerPin = board.digital[10]
middlePin = board.digital[11]
ringPin = board.digital[12]
pinkyPin = board.digital[13]

# Set pins to SERVO
thumbPin.mode = SERVO
pointerPin.mode = SERVO
middlePin.mode = SERVO
ringPin.mode = SERVO
pinkyPin.mode = SERVO

# Set Servo to 0
thumbPin.write(0)
pointerPin.write(0)
middlePin.write(0)
ringPin.write(0)
pinkyPin.write(0)

# Define MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Define Joint List
jointList = [[4, 3, 2], [8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]

# Define Function to get Angles
def getAngles(image, results, joint_list):
    
    # Loop through hands
    for hand in results.multi_hand_landmarks:
        angleList = []
        #Loop through joint sets 
        for joint in joint_list:
            a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
            b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
            c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
            
            radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
            angle = np.abs(radians*180.0/np.pi)
            
            if angle > 180.0:
                angle = 360-angle
            
            angleList.append(angle)
    return angleList

# Move servo
def moveServo(angle, pin):
    if angle > 160:
        pin.write(0)
    elif angle <= 160:
        pin.write(180)


# Computer vision loop
cap = cv2.VideoCapture(0)

# Timer for servo control (check once per second)
last_servo_update = time.time()
servo_update_interval = 1.0  # 1 second

with mpHands.Hands(max_num_hands=1, min_detection_confidence=0.8, min_tracking_confidence=0.5) as hands: 
    while cap.isOpened():
        ret, frame = cap.read()
        
        # BGR 2 RGB
        image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Flip on horizontal
        image = cv2.flip(image, 1)
        
        # Set flag
        image.flags.writeable = False
        
        # Detections
        results = hands.process(image)
        
        # Set flag to true
        image.flags.writeable = True
        
        # RGB 2 BGR
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
        
        # Rendering results
        if results.multi_hand_landmarks:
            for num, hand in enumerate(results.multi_hand_landmarks):
                mpDraw.draw_landmarks(image, hand, mpHands.HAND_CONNECTIONS, 
                                        mpDraw.DrawingSpec(color=(121, 22, 76), thickness=2, circle_radius=4),
                                        mpDraw.DrawingSpec(color=(250, 44, 250), thickness=2, circle_radius=2),
                                         )
                
                # Render left or right detection
                
            
            # Draw angles to image from joint list
            angleList = getAngles(image, results, jointList)
            
            # Only update servo once per second to reduce jitter
            for i, e in enumerate(angleList):
                moveServo(e, [thumbPin, pointerPin, middlePin, ringPin, pinkyPin][i])

            
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()