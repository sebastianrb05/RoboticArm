# Hand-Tracking Robotic Arm
A real-time hand gesture recognition system that controls a physical robotic arm using computer vision. Simply show your hand to the camera and watch as the robotic arm mimics your finger movements and hand rotation!

![Demo](demo/demo.gif)

**[📹 Watch Full Demo Video](demo/examplevideo.mp4)**


## How It's Made:

**Tech used:** Python, OpenCV, MediaPipe, Arduino, PyFirmata2, NumPy

This project combines computer vision with robotics to create an intuitive human-machine interface. Here's how it works:

**Computer Vision Pipeline:**
- Uses OpenCV to capture real-time video from your webcam
- Uses Google's MediaPipe AI models to identify 21 precise hand landmarks in real-time
- Calculates finger positions, angles, and hand rotation in 3D space

**Arduino Control System:**
- Controls 7 servo motors connected to an Arduino board via PyFirmata2
- Maps detected finger positions to servo angles (0-180 degrees)
- Implements both boolean finger control (up/down) and angle-based control
- Includes safety constraints for the thumb joint servo to prevent damage

**Key Features:**
- Real-time hand tracking with MediaPipe's machine learning models
- Individual finger control for all 5 fingers
- Hand rotation detection and mirroring
- Customizable detection confidence thresholds

## Optimizations

**Performance Improvements:**
Opted in for simple boolean servo controll instead of angles, reason for this is the mediapipe library being unstable with reading the angles between the finger joints. Resulting in jittering motors which is not optimal. There for i ended up with a more simple boolean variant checking for the fingers if they are either up or down.

## Lessons Learned:

This project was my gateway into robotics, and it taught me invaluable lessons about combining computer vision with physical hardware. The biggest technical challenge was accurately tracking finger joint angles with MediaPipe - the angle calculations were unstable and caused servo jittering, which led me to pivot to a simpler but more reliable boolean finger detection (up/down states).

This experience taught me an important engineering lesson: sometimes the elegant solution isn't the practical one. The boolean approach actually worked better for the physical constraints of the servo motors and provided smoother, more predictable movement.

I was amazed by how accessible servo control is through Arduino - what seemed like complex robotics became surprisingly straightforward with PyFirmata2. The real "wow" moment was seeing the robotic hand mirror my movements in real-time for the first time!

Working with MediaPipe showed me how powerful computer vision can be for robotics, it's exciting to think about all the other projects I could build with this foundation.

## Hardware Requirements:

- Arduino Uno/Nano (other microcontrollers would also work, but the code would need adjustments)
- 5-6 Stanard Servo Motors (MG996r)
- 1-2 Micro Servo Motors (SG90 or MG90S)
- Jumper wires and breadboard
- USB webcam
- Power supply for servos

## Installation:

1. Install required Python packages:
```bash
pip install opencv-python mediapipe pyfirmata2 numpy
```

2. Upload StandardFirmata sketch to your Arduino
3. Connect servos to digital pins 3, 8-13
4. Update COM port in code (line 12)
5. Run the program:
```bash
python cameraModual.py
```

## HandModel:
Hand Model Used For This Project: https://www.thingiverse.com/thing:2269115

I redesigned the forearm and baseplate to better suit my needs, and added an extra servo to enable independent control of the pinky and ring fingers.

![image](demo/arm.jpg)