# ASL_detection

A DNN model that can distinguish letters in ASL (American Sign Language).

# Instructions

## ASLDetection

Import the class by `from lib import ASLDetection`.

- `ad = ASLDetection()`: to initialize, you may use variable names that you prefer.
- `ad.start_video()`: it will switch on the camera on your PC or laptop in a thread. You can stop streaming by closing the window or press `ESC`.
- `ad.asl_enter()`: to allow entering alphabet in terminal for test.
- `ad.asl_judge(frame)`: `frame` needs to be a RGB picture. If you use camera to capture a frame, you may need to convert it by `frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)`. The function will return the alphabet of the frame and its mark (confidence). The alphabet would be A to Z plus 'delete' and 'space'.
## HandProcess
This class contains functions that may detect and save the position coordinate of a hand picture by using `mediapipe`.

Import the class by `from lib import HandProcess`.

- `hp = HandProcess(address_of_folder, confidence=0.5, static_mode=True)`: to initialize.
  - The files contained in `address_of_folder` should be folders that contains only pictures of the sign language of the name of them. For example, `address/A` should only contains pictures of the sign language of 'A'.
  - `confidence` will decide the accuracy of cooedinates and whether the picture can be processed.
  - `static_mode` is about whether you want to process pictures. If so, just let it be True.
- `hp.save_picture_landmarks(file_name, picture_num_of_each_alphabet=100)`: save all lamdmarks data of the sign language pictures in `address_of_folder` to a .csv file.
  - `file_name': the name of .csv file. Should end with '.csv'.
  - `picture_num_of_each_alphabet`: set the maximum number of pictures to process for each alphabet to limit performance cost.
- `hp.show_monitor()`: pop a figure that shows time consumption and number of pictures processed for each alphabet.

# About demos

## data_import_demo

A demo that shows how to import landmarks from raw pictures of sign language.

## demo

A simple demo of ASL_detection in terminal.

## tk_demo

A demo of ASL_detection with a tkinter window.

# Issues
1. Since the size of pictures uesd in training is enormous (over 1g), I only upload test landmarks csv file in /data.
2. You might need a PC with cuda to run the demo.
3. No image captured from camera is saved, so there is no threaten to privacy safety.
