import cv2
import mediapipe as mp
import os
import csv
import time
import psutil
import matplotlib.pyplot as plt

class HandProcess:
    def __init__(self,folder_path='',confidence=0.5,static_mode=True):
        self.confidence = confidence
        self.folder_path = folder_path
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(static_image_mode=static_mode, max_num_hands=1, min_detection_confidence=confidence)
        self.monitor_time = {}
        self.monitor_storage = {}
        self.monitor_picture_num = {}

    def set_static_mode(self,static_mode=True):
        self.hands = self.mp_hands.Hands(static_image_mode=static_mode, max_num_hands=1, min_detection_confidence=self.confidence)
        self.static_mode = static_mode

    def get_static_hand_landmarks(self,img_path=''):
        try:
            static_hand_landmarks = {}
            sample_img = cv2.imread(img_path)
            if sample_img is None:
                print(f"[ERROR] Picture cannot be found from {img_path}!")
                return None
            results = self.hands.process(cv2.cvtColor(sample_img, cv2.COLOR_BGR2RGB))
            #image_height, image_width, _ = sample_img.shape
            if results.multi_hand_landmarks is None:
                #print(f"[ERROR] Hand_landmarks is None! Confidence {confidence} may be too high!")
                return None
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(self.mp_hands.HandLandmark)):
                    static_hand_landmarks[self.mp_hands.HandLandmark(i).name] = [
                        hand_landmarks.landmark[self.mp_hands.HandLandmark(i).value].x,
                        hand_landmarks.landmark[self.mp_hands.HandLandmark(i).value].y,
                        hand_landmarks.landmark[self.mp_hands.HandLandmark(i).value].z
                    ]
        except Exception as e:
            print(f"[Error] When getting static landmarks, {e} occured!")
        else:
            return static_hand_landmarks
            
    def save_pictures_landmarks(self,file_name,picture_num_of_each_alphabet=100):
        # sourcery skip: remove-dict-keys
        with open(f'./data/{file_name}.csv','w',newline='',encoding='utf-8') as f:
            csv_write = csv.writer(f)
            hand_label = self.get_hand_label()
            csv_write.writerow(hand_label)
            for alphabet in os.scandir(self.folder_path):
                if alphabet.name == 'nothing':
                    continue      
                
                time0 = time.time()

                alphabet_landmarks = self.import_alphabet_landmarks(alphabet.name,picture_num_of_each_alphabet)
                
                #print(u'Time usage: %.2f s' % ((time.time()-time0)))
                self.monitor_time[alphabet.name] = time.time()-time0
                #print(u'memory usage: %.4f GB' % (psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024) )
                self.monitor_storage[alphabet.name] = psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024 / 1024

                csv_write.writerows(alphabet_landmarks)
        print("[INFO] Picture saved successfully!")

    def import_alphabet_landmarks(self,alphabet,picture_num_of_each_alphabet=100):
        # sourcery skip: remove-dict-keys
        landmarks = []
        try:
            max_scan_num = max(100, picture_num_of_each_alphabet)
            picture_num = 0
            scan_num = 0
            for picture in os.scandir(os.path.join(self.folder_path,alphabet)):
                scan_num = scan_num + 1
                if scan_num > max_scan_num:
                    #print(f"[INFO] {picture_num} pictures of {alphabet} saved! {scan_num-1} pictures scanned!")
                    self.monitor_picture_num[alphabet] = picture_num
                    break
                hand_landmarks = self.get_static_hand_landmarks(picture.path)
                if hand_landmarks is None:
                    continue
                picture_num = picture_num + 1
                if picture_num > picture_num_of_each_alphabet:
                    #print(f"[INFO] {picture_num-1} pictures of {alphabet} saved successfully!")
                    self.monitor_picture_num[alphabet] = picture_num - 1
                    break
                _, hand_vec = self.hand_landmarks_to_vector(hand_landmarks)
                hand_vec.append(self.alphabet_to_long(alphabet))
                landmarks.append(hand_vec)
        except Exception as e:
            print(f"[ERROR] An error occured when import picture:{picture.name} : {e}")
            quit()
        else:
            return landmarks

    def hand_landmarks_to_vector(self,static_hand_landmarks):
        landmarks_label = []
        landmarks_vector = []
        cartisian_label = ['X','Y','Z']
        for label,landmarks_position in static_hand_landmarks.items():
            landmarks_vector.extend(iter(landmarks_position))
            landmarks_label.extend(f'{label}_{cartisian}' for cartisian in cartisian_label)
        return landmarks_label, landmarks_vector

    def alphabet_to_long(self,alphabet:str):
        dictionary = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',"P","Q","R","S","T","U","V","W","X","Y","Z","del","space"]
        try:
            index = dictionary.index(alphabet)
        except ValueError:
            print(f"[ERROR] Can't find alphabet {alphabet}!")
            return None
        except Exception as e:
            print(f"[ERROR] {e}!")
            return None
        else:
            return index
        
    def long_to_alphabet(self,number):
        dictionary = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',"P","Q","R","S","T","U","V","W","X","Y","Z","del","space"]
        return dictionary[number]
        
    def get_hand_label(self):
        img_path = r'.\pictures\asl_alphabet_train\asl_alphabet_train\A\A1.jpg'
        hand_landmarks = self.get_static_hand_landmarks(img_path)
        hand_label, _ = self.hand_landmarks_to_vector(hand_landmarks)
        hand_label.append('result')
        return hand_label

    def show_monitor(self):

        plt.rc('font',family='Times New Roman', size=10)
        plt.rcParams["axes.unicode_minus"]=False
        plt.figure(figsize=(12, 6))

        x_data = list(self.monitor_time.keys())
        x_pos = list(range(len(x_data)))
        y1_data = list(self.monitor_picture_num.values())
        y2_data = list(self.monitor_storage.values())
        y3_data = list(self.monitor_time.values())

        plt.suptitle("Monitor information of import")

        plt.subplot(3,1,1)
        plt.bar(x_pos,y1_data,lw=0.5,fc="r",width=0.5,label="Picture number")
        plt.xticks(range(len(x_data)),x_data)
        for i,j in zip(x_pos,y1_data):
            plt.text(i,j,"%d"%j,ha="center",va="bottom")
        plt.xlabel("alphabet")
        plt.ylabel("number (#)")

        plt.subplot(3,1,2)
        plt.plot(x_pos,y2_data,lw=1,color='g',linestyle='-',marker='.',label="Memory usage")
        plt.xticks(range(len(x_data)),x_data)
        for i,j in zip(x_pos,y2_data):
            plt.text(i,j,"%.4f"%j,ha="center",va="bottom")
        plt.xlabel("alphabet")
        plt.ylabel("memory usage (GB)")

        plt.subplot(3,1,3)
        plt.bar(x_pos,y3_data,lw=0.5,fc="b",width=0.5,label="Time usage")
        plt.xticks(range(len(x_data)),x_data)
        for i,j in zip(x_pos,y3_data):
            plt.text(i,j,"%.2fs"%j,ha="center",va="bottom")
        plt.xlabel("alphabet")
        plt.ylabel("time usage (s)")

        plt.tight_layout()

        plt.show()