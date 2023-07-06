'''
A demo that shows how to import landmarks from raw pictures of sign language.
'''

from lib.hand_landmark_trans import HandProcess

try:
    hp = HandProcess(r'.\pictures\asl_alphabet_train',0.7)
    hp.save_pictures_landmarks('landmarks_test',10)
    hp.show_monitor()
except Exception as e:
    print(e)