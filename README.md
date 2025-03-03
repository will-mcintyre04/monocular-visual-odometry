# Monocular Visual Odometery

Simple visual odometry system for a monocular camera using ORB feature detection and FLANN-based matching for matching keypoints between consecutive frames. The estimated path of the camera is compared to ground truth poses (from the KITTI dataset) to compare and calculate drift error.

Built using Python and the OpenCV library, and bokeh for plotting.

Currently in development for a real time, live system using the webcam.

![image](https://github.com/user-attachments/assets/ab4532bd-da84-4b45-9a76-400b0316d0dc)
![Screenshot 2025-03-02 215612](https://github.com/user-attachments/assets/fb5a9762-cae6-48ed-b88a-d87fdad4aaf6)


Big thanks for all the support found online. A few helpful sources I used:
- https://www.youtube.com/@NicolaiAI
- https://github.com/Shiaoming/Python-VO
- https://docs.opencv.org/3.4/d1/d89/tutorial_py_orb.html
