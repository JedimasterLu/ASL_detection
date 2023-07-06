import cv2
import torch
import mediapipe as mp
from lib.DNN import Net
import threading
import time

class ASLDetection():
    def __init__(self):
        self.cap = cv2.VideoCapture(0)

        self.model = Net(n_feature=63, n_hidden=128, n_output=28)
        self.model.load_state_dict(torch.load("./data/net_parameter.pkl"))
        self.model.eval()

        self.mp_drawing = mp.solutions.drawing_utils
        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.7,
            min_tracking_confidence=0.7)
        
        self.detected_alphabet = ''
        self.entered_msg = ''

    def start_video(self):
        cap = cv2.VideoCapture(0)
        threads = threading.Thread(target=self.asl_enter,name='asl_enter',daemon=True)
        threads.start()
        while True:
            _,frame = self.cap.read()
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.detected_alphabet,mark = self.asl_judge(frame)
            frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            if self.detected_alphabet:      
                cv2.putText(frame,self.detected_alphabet,(50,100),0,1.3,(0,0,255),2)
                cv2.putText(frame, f'{str(mark)}%', (50,150), 0, 1.3, (0,0,255), 2)
            cv2.imshow('ASL_Detection Demo', frame)
            key = cv2.waitKey(10)
            if key == 27 or cv2.getWindowProperty('ASL_Detection Demo', cv2.WND_PROP_VISIBLE) < 1.0: 
                break
        cap.release()
        cv2.destroyAllWindows()
        return self.entered_msg
        
    def asl_enter(self):
        print("asl_enter begins:")

        self.entered_msg = ''
        while True:
            alphabet = self.detected_alphabet
            time.sleep(1)
            if self.detected_alphabet == alphabet and self.detected_alphabet == 'space':
                self.entered_msg = f"{self.entered_msg} "
                print(f'Entered! Now is:{self.entered_msg}')
            elif self.detected_alphabet == alphabet and self.detected_alphabet == 'del':
                self.entered_msg = self.entered_msg[:-1]
                print(f'Deleted! Now is:{self.entered_msg}')
            elif self.detected_alphabet == alphabet and self.detected_alphabet != '':
                self.entered_msg = self.entered_msg + alphabet
                print(f'Spaced! Now is:{self.entered_msg}')
            else:
                continue
    
    def asl_judge(self,frame):
        results = self.hands.process(frame)
        alphabet = ''
        mark = 0
        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                hand_local = []
                for i in range(21):
                    x = float(hand_landmarks.landmark[i].x)
                    y = float(hand_landmarks.landmark[i].y)
                    z = float(hand_landmarks.landmark[i].z)
                    hand_local.extend((x, y, z))
            logit = self.model(torch.tensor(hand_local,dtype=torch.float32).cuda())
            soft_outputs = torch.nn.functional.softmax(logit, dim=0) #pass through softmax
            top_p, top_class = soft_outputs.topk(1, dim=0) # select top probability as prediction
            mark = round(float(top_p.cpu().detach().numpy())*100,2)
            alphabet = self.long_to_alphabet(int(top_class.cpu().detach().numpy()))
        return alphabet, mark
    
    def long_to_alphabet(self,number):
        dictionary = ['A','B','C','D','E','F','G','H','I','J','K','L','M','N','O',"P","Q","R","S","T","U","V","W","X","Y","Z","del","space"]
        return dictionary[number]
