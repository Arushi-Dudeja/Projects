**Inspiration**
Drowsy Driving is a significant factor in a huge proportion of road accidents globally

**What it does**
We present a simple real-time video monitoring application to alert drivers as they turn sleepy

**How we built it**
We used OpenCV to monitor eyes of the driver and python libraries pygame to send sound alerts to wake up the driver. We used acknowledgement to confirm the driver is up, if not we used twilio to automatically call 911 for emergency

**Challenges we ran into**
Integrating the python script to an actual web application was time-consuming

**Accomplishments that we're proud of**
Figuring out a way to automatically call a phone number through python API

**What we learned**
OpenCV for computer vision, CMake tools for development, Python API for calling, Figma for interactive design

**What's next for Wake The Beep Up**
1. While this is just a way for us to put our step in the game, we have several ideas for the future to enhance our application:
2. Adding visual alerts to pop up/flash the phone screen
3. Adding haptic alerts in the driver seat by integrating hardware devices to create vibrations for alerting
4. Confirming state of being awake through solving a quick math question
5. Adding option to either call a cab in the application
