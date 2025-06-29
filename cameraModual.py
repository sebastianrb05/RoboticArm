import os
os.environ["OPENCV_VIDEOIO_MSMF_ENABLE_HW_TRANSFORMS"] = "0"
import cv2
import numpy as np
import time
import mediapipe as mp
from pyfirmata2 import Arduino, SERVO
import math


# Define Arduino
board = Arduino('COM3')

# Define Servo
thumbPin = board.digital[8]
pointerPin = board.digital[9]
middlePin = board.digital[10]
ringPin = board.digital[11]
pinkyPin = board.digital[12]
thumbJoint = board.digital[13]

rotationPin = board.digital[3]

# Set pins to SERVO
thumbPin.mode = SERVO
thumbJoint.mode = SERVO
pointerPin.mode = SERVO
middlePin.mode = SERVO
ringPin.mode = SERVO
pinkyPin.mode = SERVO
rotationPin.mode = SERVO

# Set Servo to 0
thumbPin.write(0)
pointerPin.write(0)
middlePin.write(0)
ringPin.write(0)
pinkyPin.write(0)
#Thumb joint servo should never be out of the interval 180-90 degrees !!!
thumbJoint.write(180)
rotationPin.write(0)

# Define MediaPipe Hands
mpHands = mp.solutions.hands
hands = mpHands.Hands(max_num_hands=1, min_detection_confidence=0.7)
mpDraw = mp.solutions.drawing_utils

# Define Joint List

tipIds = [4, 8, 12, 16, 20] # Thumb, Index, Middle, Ring, Pinky
dipIds = [3, 7, 11, 15, 19]
pipIds = [2, 6, 10, 14, 18]

# Move servo
def moveServoAngle(angle, pin):
    if angle > 160:
        pin.write(0)
    elif angle <= 160:
        pin.write(180)

def moveServoBool(bool, pin):
    if bool:
        pin.write(180)
    else:
        pin.write(0)



# HandDetector class
class HandDetector:
    def __init__(self, mode=False, maxHands=2, detectionCon=0.5, trackCon=0.5):
        self.mode = mode
        self.maxHands = maxHands
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpHands = mp.solutions.hands
        self.hands = self.mpHands.Hands(
            static_image_mode=self.mode,
            max_num_hands=self.maxHands,
            min_detection_confidence=self.detectionCon,
            min_tracking_confidence=self.trackCon
        )
        self.mpDraw = mp.solutions.drawing_utils

        # Finger tip landmark IDs
        self.tipIds = [4, 8, 12, 16, 20]
        self.dipIds = [3, 7, 11, 15, 19]  # Thumb, Index, Middle, Ring, Pinky
        self.pipIds = [2, 6, 10, 14, 18]
        self.jointList = [[4, 3, 2], [8, 7, 6], [12, 11, 10], [16, 15, 14], [20, 19, 18]]
        self.landmarks = []
    

    def findHands(self, img, draw=True):
        """Detect hands and optionally draw landmarks"""
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.hands.process(imgRGB)

        if self.results.multi_hand_landmarks:
            for handLms in self.results.multi_hand_landmarks:
                if draw:
                    self.mpDraw.draw_landmarks(img, handLms, self.mpHands.HAND_CONNECTIONS)
        return img

    def findPosition(self, img, handNo=0, draw=True):
        """Get landmark positions for a specific hand"""
        self.landmarks = []
        if self.results.multi_hand_landmarks:
            if handNo < len(self.results.multi_hand_landmarks):
                myHand = self.results.multi_hand_landmarks[handNo]
                for id, lm in enumerate(myHand.landmark):
                    h, w, c = img.shape
                    cx, cy = int(lm.x * w), int(lm.y * h)
                    self.landmarks.append([id, cx, cy])
                    if draw:
                        cv2.circle(img, (cx, cy), 5, (255, 0, 255), cv2.FILLED)
        return self.landmarks
    
    def getDistance(self, id1, id2):
        return math.sqrt((self.landmarks[id1][1]-self.landmarks[id2][1])**2 + (self.landmarks[id1][2]-self.landmarks[id2][2])**2)

    def fingersUp(self):
        """
        Custom fingersUp function - returns list of which fingers are up
        Returns: [thumb, index, middle, ring, pinky] where 1=up, 0=down
        """
        fingers = []
        
        


        if len(self.landmarks) != 0:
            # Thumb - compare x coordinates (horizontal movement)
            # For right hand: thumb up if tip is to the right of previous joint
            # This is a simplified version - you might want to add handedness detection
            
            # Thumb - compare distance from wrist
            # Thumb is UP if tip is farther from wrist than DIP joint
            thumbTipDistance = self.getDistance(self.tipIds[0], 0)
            thumbDipDistance = self.getDistance(self.dipIds[0], 0)
            if thumbTipDistance > thumbDipDistance:
                fingers.append(0)  # Thumb is UP
            else:
                fingers.append(1)  # Thumb is DOWN

            # Four fingers - finger is UP if tip is farther from wrist than both DIP and PIP
            for id in range(1, 5):
                tipDistance = self.getDistance(self.tipIds[id], 0)
                dipDistance = self.getDistance(self.dipIds[id], 0)
                pipDistance = self.getDistance(self.pipIds[id], 0)
                
                if tipDistance > dipDistance and tipDistance > pipDistance:
                    fingers.append(0)  # Finger is UP
                else:
                    fingers.append(1)  # Finger is DOWN
        
        return fingers
    
    def getAngles(self, image, results, jointList):
        # Loop through hands
        for hand in results.multi_hand_landmarks:
            angleList = []
            #Loop through joint sets 
            for joint in self.jointList:
                a = np.array([hand.landmark[joint[0]].x, hand.landmark[joint[0]].y]) # First coord
                b = np.array([hand.landmark[joint[1]].x, hand.landmark[joint[1]].y]) # Second coord
                c = np.array([hand.landmark[joint[2]].x, hand.landmark[joint[2]].y]) # Third coord
                
                radians = np.arctan2(c[1] - b[1], c[0]-b[0]) - np.arctan2(a[1]-b[1], a[0]-b[0])
                angle = np.abs(radians*180.0/np.pi)
                
                if angle > 180.0:
                    angle = 360-angle
                
                angleList.append(angle)
        return angleList

    def calculate_hand_rotation(self):
        """Calculate hand rotation angle - returns 0 if no hand detected"""
        if len(self.landmarks) < 18:  # Need at least 18 landmarks for calculation
            return 0
            
        # Get key points for rotation calculation (landmarks are [id, x, y])
        wrist = self.landmarks[0]  # [0, x, y]
        middle_mcp = self.landmarks[9]  # [9, x, y]  
        index_mcp = self.landmarks[5]   # [5, x, y]
        pinky_mcp = self.landmarks[17]  # [17, x, y]
        
        # Calculate vectors using 2D coordinates (x, y)
        # Vector from wrist to middle finger base
        palm_vector = np.array([
            middle_mcp[1] - wrist[1],  # x difference
            middle_mcp[2] - wrist[2]   # y difference
        ])
        
        # Vector across the knuckles (index to pinky)
        knuckle_vector = np.array([
            pinky_mcp[1] - index_mcp[1],  # x difference
            pinky_mcp[2] - index_mcp[2]   # y difference
        ])
        
        # Calculate rotation angle from knuckle vector
        roll = math.atan2(knuckle_vector[1], knuckle_vector[0]) * 180 / math.pi
        
        # Normalize to 0-180 degrees for servo control
        if roll < 0:
            roll += 360
        if roll > 180:
            roll = 360 - roll
            
        # Map to servo range (0-180)
        roll = max(0, min(180, roll))
        
        return roll




# Hand rotation function
# TODO: Implement hand rotation function
# Thoughts: check for distance in deapth between thumb and pinky, to establish partial rotation
# or the width of the hand from side to side, and establish if back or front of the hand to see what way to rotate   
# or i could make a CNN predicting the hand rotation and pair that with the servos from paper i found online
# TODO: add left and right hand detection, and then use that to determine the rotation



cap = cv2.VideoCapture(0)
detector = HandDetector(maxHands=1)

# Timer for servo control (check once per second)
last_servo_update = time.time()
servo_update_interval = 1.0  # 1 second

# For FPS calculation
pTime = 0

while cap.isOpened():
    success, frame = cap.read()

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
    

    # Find hands# Find hands
    img = detector.findHands(frame)
    lmList = detector.findPosition(frame, draw=False)

    # Rendering results
    # Draw angles to image from joint list
    #angleList = detector.getAngles(image, results, detector.jointList)
    
    # uses the angle list to move the servos
    ''' Only update servo once per second to reduce jitter
    for i, e in enumerate(angleList):
        moveServoAngle(e, [thumbPin, pointerPin, middlePin, ringPin, pinkyPin][i])
        # if thumbpin is at max position (1.0 = 180 degrees), then thumbjoint should be 90
        if thumbPin.read() >= 0.95:  # Near maximum position
            thumbJoint.write(90)
        else:
            thumbJoint.write(180)
    '''   

    # Using the fingersUp function to move the servos
    fingers = detector.fingersUp()
    for i, e in enumerate(fingers):
        moveServoBool(e, [thumbPin, pointerPin, middlePin, ringPin, pinkyPin][i])

        # if thumbpin is at max position (1.0 = 180 degrees), then thumbjoint should be 90
        if thumbPin.read() >= 0.95:  # Near maximum position
            thumbJoint.write(90)
        else:
            thumbJoint.write(180)

    
    # Hand rotation (use the first detected hand)
    hand_rotation = detector.calculate_hand_rotation()
    # set rotation pin to hand rotation
    rotationPin.write(hand_rotation)


   
    if len(lmList) != 0:
        # Get fingers up status
        fingers = detector.fingersUp()
        
        totalFingers = fingers.count(1)
        
        # Display finger count
        cv2.putText(img, f'Fingers: {totalFingers}', (10, 70), 
                    cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 0), 3)
        
    
    # Calculate and display FPS
    cTime = time.time()
    fps = 1 / (cTime - pTime)
    pTime = cTime
    cv2.putText(img, f'FPS: {int(fps)}', (400, 70), 
                cv2.FONT_HERSHEY_PLAIN, 3, (255, 0, 255), 3)
    
    # Show image
    cv2.imshow('Hand Tracking', img)
    
    # Break on 'q' key press
    if cv2.waitKey(10) & 0xFF == ord('q'):
        break
    
cap.release()
cv2.destroyAllWindows()

'''

# TODO: override finger angles with model prediction from googles mediapipe model
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
                # if thumbpin is at max position (1.0 = 180 degrees), then thumbjoint should be 90
                if thumbPin.read() >= 0.95:  # Near maximum position
                    thumbJoint.write(90)
                else:
                    thumbJoint.write(180)
            
            # Hand rotation (use the first detected hand)
            first_hand = results.multi_hand_landmarks[0]
            hand_rotation = calculate_hand_rotation(first_hand.landmark)
            # set rotation pin to hand rotation
            rotationPin.write(hand_rotation)


            
        # Save our image    
        #cv2.imwrite(os.path.join('Output Images', '{}.jpg'.format(uuid.uuid1())), image)
        cv2.imshow('Hand Tracking', image)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
'''