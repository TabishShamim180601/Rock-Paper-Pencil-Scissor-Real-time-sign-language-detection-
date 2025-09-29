import mediapipe as mp
import cv2 
import os
import matplotlib.pyplot as plt 
import pickle

mp_hands = mp.solutions.hands 
mp_drawing = mp.solutions.drawing_utils 
mp_drawing_styles = mp.solutions.drawing_styles

hands = mp_hands.Hands(static_image_mode=True,min_detection_confidence=0.3)


DATA_DIR = 'images' #the directory where the images collected from webcam are stored

data = []   #to store features
labels = [] #to store labels

for dir_ in os.listdir(DATA_DIR): #iterating over each directory in images
    for img_path in os.listdir(os.path.join(DATA_DIR,dir_)):
        data_aux = []
        img = cv2.imread(os.path.join(DATA_DIR,dir_,img_path)) #reading the image
        img_rgb = cv2.cvtColor(img,cv2.COLOR_BGR2RGB) #converting to rgb

        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                for i in range(len(hand_landmarks.landmark)):
                    x = hand_landmarks.landmark[i].x #x coordinate of image
                    y = hand_landmarks.landmark[i].y #y coordinate of image
                    data_aux.append(x)
                    data_aux.append(y)
            
            data.append(data_aux)
            labels.append(dir_)

f = open('data.pickle','wb')
pickle.dump({'data':data,'labels':labels},f)
f.close()