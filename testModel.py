import numpy as np
import torch
from grabscreen import grab_screen
import cv2
import time
from directkeys import PressKey,ReleaseKey, W, A, S, D
from ResNet import resnet34
from getkeys import key_check
import random

WIDTH = 160
HEIGHT = 120
LR = 1e-3
MODEL_NAME = ''

w = [1,0,0,0,0,0,0,0,0]
s = [0,1,0,0,0,0,0,0,0]
a = [0,0,1,0,0,0,0,0,0]
d = [0,0,0,1,0,0,0,0,0]
wa = [0,0,0,0,1,0,0,0,0]
wd = [0,0,0,0,0,1,0,0,0]
sa = [0,0,0,0,0,0,1,0,0]
sd = [0,0,0,0,0,0,0,1,0]
nk = [0,0,0,0,0,0,0,0,1]

def straight():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(D)
    ReleaseKey(S)

def left():
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)
    ReleaseKey(S)

def right():
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(S)
    
def reverse():
    PressKey(S)
    ReleaseKey(A)
    ReleaseKey(W)
    ReleaseKey(D)


def forward_left():
    PressKey(W)
    PressKey(A)
    ReleaseKey(D)
    ReleaseKey(S)
    
    
def forward_right():
    PressKey(W)
    PressKey(D)
    ReleaseKey(A)
    ReleaseKey(S)

    
def reverse_left():
    PressKey(S)
    PressKey(A)
    ReleaseKey(W)
    ReleaseKey(D)

    
def reverse_right():
    PressKey(S)
    PressKey(D)
    ReleaseKey(W)
    ReleaseKey(A)

def no_keys():
    PressKey(W)
    ReleaseKey(A)
    ReleaseKey(S)
    ReleaseKey(D)
    


model = resnet34()
model_weight_path = './ResNet34.pth'
model.load_state_dict(torch.load(model_weight_path),strict = False)
model.eval()
def main():
    for i in list(range(4))[::-1]:
        print(i+1)
        time.sleep(1)

    paused = False
    while(True):
        if not paused:
            # 800x600 windowed mode
            screen = grab_screen(region=(0,40,800,640))
            screen = cv2.cvtColor(screen, cv2.COLOR_BGR2GRAY)
            screen = cv2.resize(screen, (WIDTH,HEIGHT))
            screen = np.array(list(screen)).reshape(-1,1,WIDTH,HEIGHT)
            screen = torch.from_numpy(screen)
            screen = screen.type(torch.FloatTensor)
            prediction = model(screen)
            prediction = torch.max(prediction,dim = 1)[1]
            if prediction.item() == np.argmax(w):
                straight()              
            elif prediction.item() == np.argmax(s):
                reverse()
            if prediction.item() == np.argmax(a):
                left()
            if prediction.item() == np.argmax(d):
                right()
            if prediction.item() == np.argmax(wa):
                forward_left()
            if prediction.item() == np.argmax(wd):
                forward_right()
            if prediction.item() == np.argmax(sa):
                reverse_left()
            if prediction.item() == np.argmax(sd):
                reverse_right()
            if prediction.item() == np.argmax(nk):
                no_keys()
            
        keys = key_check()

        # p pauses game and can get annoying.
        if 'T' in keys:
            if paused:
                paused = False
                print('start...')
                time.sleep(1)
            else:
                paused = True
                print('pause...')
                ReleaseKey(A)
                ReleaseKey(W)
                ReleaseKey(D)
                time.sleep(1)

main()       
