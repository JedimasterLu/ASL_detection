'''
A demo of ASL_detection with a tkinter window.
'''

import cv2
import tkinter as tk
from tkinter import *
from PIL import Image, ImageTk
from lib import ASLDetection
import threading
import time
import keyboard

cap = cv2.VideoCapture(0)
asl = ASLDetection()

root = tk.Tk()

root.resizable(False, False)
root.wm_attributes('-topmost',1)
root.overrideredirect(True)

screen_width = root.winfo_screenwidth()
screen_height = root.winfo_screenheight()

root.title('ASL')
root.geometry(f'300x250+{screen_width-310}+{screen_height-460}')

root_close_flg = False

def close_root():
    global root_close_flg
    root_close_flg = True

image_width = 300
image_height = 225

fr1 = Frame(root,width=300,height=225)
fr1.pack(side='top')
fr2 = Frame(root,width=300,height=25)
fr2.pack(side="top")

canvas = Canvas(fr1,bg='grey',width=image_width,height=image_height,highlightthickness=0)
text = Entry(fr2,font=("Helvetica",14),state='normal')
close_button = Button(fr2,text="Close",font=("Helvetica",12),bg='grey',command=close_root,width=25)
canvas.pack(fill='both')
text.pack(side='left',fill='both')
close_button.pack(side='right',fill='y')

current_input = ''
current_alphabet = ''
current_mark = 0
enter_lst = []

def asl_enter():
    global current_input,text,root,current_alphabet,current_mark,enter_lst
    current_input = ""
    last_alphabet = ''
    while True:
        if current_mark > 90:
            last_alphabet = current_alphabet
        if current_alphabet == 'del':
            time.sleep(1.2)
        else:
            time.sleep(1)
        if last_alphabet == current_alphabet and current_mark > 90:
            if current_alphabet == '':
                continue
            elif current_alphabet == 'del':
                current_input = current_input[:-1]
            elif current_alphabet == "space":
                current_input += ' '
            else:
                current_input += current_alphabet
        if len(current_input) > 2 and current_input[-1] == ' ' and current_input[-2] == ' ':
            enter_lst.append(current_input[:-2])
            current_input = ''

def transfer_frame(frame):  # sourcery skip: inline-immediately-returned-variable
    cvimage = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
    pilImage=Image.fromarray(cvimage)
    pilImage = pilImage.resize((image_width, image_height),Image.LANCZOS)
    image = ImageTk.PhotoImage(image=pilImage)
    return image

threading_start_flg = False

def run():
    global cap, root, image, text
    global current_alphabet, current_mark, threading_start_flg, enter_lst

    if threading_start_flg == False:
        t = threading.Thread(target=asl_enter,name='asl_enter',daemon=True)
        t.start()
        threading_start_flg = True

    ref,frame=cap.read()

    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    alphabet,mark = asl.asl_judge(frame)
    current_mark = mark
    current_alphabet = alphabet
    frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
    if alphabet:
        cv2.putText(frame,alphabet,(50,100),0,1.3,(0,0,255),2)
        cv2.putText(frame,f'{str(mark)}%',(50,150),0,1.3,(0,0,255),2)

    image = transfer_frame(frame)
    canvas.create_image(0,0,anchor='nw',image=image)
    canvas.update()

    text.delete(0,"end")
    text.insert(END, f'"{current_input}"')
    text.update()

    while enter_lst:
        enter_msg = enter_lst.pop()
        keyboard.write(enter_msg)

    if root_close_flg == False:
        root.after(1,run)
    else:
        root.destroy()

root.after(1,run)
root.mainloop()

cap.release()