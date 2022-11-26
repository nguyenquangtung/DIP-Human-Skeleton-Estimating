#%%
#Group 4
#Nguyễn Quang Tùng 	19110141
#contact me via gmail: tt.quangtung.ld@gmail.com
from tkinter import *
from tkinter.filedialog import askopenfilename
from PIL import Image, ImageTk
from tkinter import filedialog
import cv2 as cv2
import numpy as np 
from numpy import asarray
from matplotlib import pyplot as plt
import os
import time
from cvzone.FaceDetectionModule import FaceDetector
from cvzone.HandTrackingModule import HandDetector
from cvzone.FaceMeshModule  import FaceMeshDetector
import mediapipe as mp
import math
import argparse

class PoseDetector:
    """
    Estimates Pose points of a human body using the mediapipe library.
    """

    def __init__(self, mode=False, smooth=True,
                 detectionCon=0.5, trackCon=0.5):

        self.mode = mode
        self.smooth = smooth
        self.detectionCon = detectionCon
        self.trackCon = trackCon

        self.mpDraw = mp.solutions.drawing_utils
        self.mpPose = mp.solutions.pose
        self.pose = self.mpPose.Pose(static_image_mode=self.mode,
                                     smooth_landmarks=self.smooth,
                                     min_detection_confidence=self.detectionCon,
                                     min_tracking_confidence=self.trackCon)

    def findPose(self, img, draw=True):
 
        imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        self.results = self.pose.process(imgRGB)
        if self.results.pose_landmarks:
            if draw:
                self.mpDraw.draw_landmarks(img, self.results.pose_landmarks,
                                           self.mpPose.POSE_CONNECTIONS)
        return img

    def findPosition(self, img, draw=True, bboxWithHands=False):
        self.lmList = []
        self.bboxInfo = {}
        if self.results.pose_landmarks:
            for id, lm in enumerate(self.results.pose_landmarks.landmark):
                h, w, c = img.shape
                cx, cy, cz = int(lm.x * w), int(lm.y * h), int(lm.z * w)
                self.lmList.append([id, cx, cy, cz])

            # Bounding Box
            ad = abs(self.lmList[12][1] - self.lmList[11][1]) // 2
            if bboxWithHands:
                x1 = self.lmList[16][1] - ad
                x2 = self.lmList[15][1] + ad
            else:
                x1 = self.lmList[12][1] - ad
                x2 = self.lmList[11][1] + ad

            y2 = self.lmList[29][2] + ad
            y1 = self.lmList[1][2] - ad
            bbox = (x1, y1, x2 - x1, y2 - y1)
            cx, cy = bbox[0] + (bbox[2] // 2), \
                     bbox[1] + bbox[3] // 2

            self.bboxInfo = {"bbox": bbox, "center": (cx, cy)}

            if draw:
                cv2.rectangle(img, bbox, (255, 0, 255), 3)
                cv2.circle(img, (cx, cy), 5, (255, 0, 0), cv2.FILLED)

        return self.lmList, self.bboxInfo


#% Create a window
root = Tk()
root.title('Group 4 finalterm project')
var = StringVar()  
label = Label(root, textvariable = var, bg = "WHITE",font=(None, 13), bd = 10, justify = CENTER, padx = 5, pady = 5)  
var.set("ESTIMATE HUMAN SKELETON FROM IMAGE /VIDEO \n Nguyễn Quang Tùng 1911014\n Chu Nguyễn Hoàng Sơn 19119128") 
label.pack()


     #%Crate a frame1 : process image
frame1=LabelFrame(root,text='Image')
frame1.pack( fill=BOTH, side=LEFT ,padx=5,pady=5,ipadx=5,ipady=5)
     #%Crate a frame2 : estimate skeleton
frame2=LabelFrame(root,text='ESTIMATE SKELETON')
frame2.pack( fill=BOTH, side=RIGHT ,padx=5,pady=5,ipadx=5,ipady=5)
     #%Crate a frame2 : extend functions
frame3=LabelFrame(root,text='EXTEND FUNCTIONS')
frame3.pack( fill=BOTH,side=RIGHT  ,padx=5,pady=5,ipadx=5,ipady=5)
#% create a label
my_label1 = Label(frame1,text='Original')
my_label1.pack(side=LEFT)

# my_label2 = Label(frame1,text='Estimated skeleton')
# my_label2.pack(side=RIGHT)

#% Browse button

    
def denoise(frame):
    frame = cv2.medianBlur(frame,11)
    frame = cv2.GaussianBlur(frame,(11,11),0)
    return frame

def combinationetm():
    pTime = 0
    detector=PoseDetector()
    detector2= HandDetector(detectionCon=0.8, maxHands=2)
    detector3 = FaceDetector()
    cap=cv2.VideoCapture(0)
    while True:
        success,img=cap.read()
        img=detector.findPose(img)
        lmList,bboxInfo=detector.findPosition(img,bboxWithHands=True)
        hands, img = detector2.findHands(img) 
        if hands:
         # Hand 1
             hand1 = hands[0]
             lmList1 = hand1["lmList"]  # List of 21 Landmark points
             bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
             centerPoint1 = hand1['center']  # center of the hand cx,cy
             handType1 = hand1["type"]  # Handtype Left or Right

             fingers1 = detector2.fingersUp(hand1)
             
        img, bboxs = detector3.findFaces(img)

        if bboxs:
        # bboxInfo - "id","bbox","score","center"
            center = bboxs[0]["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
        cTime = time.time() # frame rate
        fps_rate = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps_rate)}', (20, 50), cv2.FONT_HERSHEY_PLAIN,3,(0, 255, 0), 4)
        cv2.imshow("Estimate Skeleton",img)
    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            cap.release() 
            break
        
def facedect():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceDetector()

    while True:
        success, img = cap.read()
        img, bboxs = detector.findFaces(img)

        if bboxs:
        # bboxInfo - "id","bbox","score","center"
            center = bboxs[0]["center"]
            cv2.circle(img, center, 5, (255, 0, 255), cv2.FILLED)
        cTime = time.time() # frame rate
        fps_rate = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps_rate)}', (20, 50), cv2.FONT_HERSHEY_PLAIN,3,(0, 255, 0), 4)
        cv2.imshow("face detection", img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            cap.release() 
            break

def handtrk():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = HandDetector(detectionCon=0.8, maxHands=2)
    while True:
       # Get image frame
       success, img = cap.read()
       # Find the hand and its landmarks
       hands, img = detector.findHands(img)  # with draw
       # hands = detector.findHands(img, draw=False)  # without draw

       if hands:
        # Hand 1
            hand1 = hands[0]
            lmList1 = hand1["lmList"]  # List of 21 Landmark points
            bbox1 = hand1["bbox"]  # Bounding box info x,y,w,h
            centerPoint1 = hand1['center']  # center of the hand cx,cy
            handType1 = hand1["type"]  # Handtype Left or Right

            fingers1 = detector.fingersUp(hand1)
        # Display
       cTime = time.time() # frame rate
       fps_rate = 1 / (cTime-pTime)
       pTime = cTime
       cv2.putText(img, f'FPS: {int(fps_rate)}', (20, 50), cv2.FONT_HERSHEY_PLAIN,3,(0, 255, 0), 4)
       cv2.imshow("hand tracking", img)
       k = cv2.waitKey(30) & 0xff
       if k == 27:
           cv2.destroyAllWindows()
           cap.release() 
           break
def facemsh():
    pTime = 0
    cap = cv2.VideoCapture(0)
    detector = FaceMeshDetector(maxFaces=2)
    while True:
        success, img = cap.read()
        img, faces = detector.findFaceMesh(img)
        if faces:
            print(faces[0])
        # Display
        cTime = time.time() # frame rate
        fps_rate = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps_rate)}', (20, 50), cv2.FONT_HERSHEY_PLAIN,3,(0, 255, 0), 4)
        cv2.imshow("facemesh detection", img)
        k = cv2.waitKey(30) & 0xff
        if k == 27:
           cv2.destroyAllWindows()
           cap.release() 
           break  

def estmimg():
    global img
    link = askopenfilename()
    img = ImageTk.PhotoImage(Image.open(link))
    #my_img2=cv2.resize(my_img2,(800, 800))
    my_label1.configure(image=img)
    my_label1.image = img
    # image_array = np.array(img)
    # pil_image=Image.fromarray(image_array)
    img = cv2.imread(link,0)
    im_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    detector=PoseDetector()
    img=detector.findPose(im_rgb)
    lmList,bboxInfo=detector.findPosition(img,bboxWithHands=True)
    cv2.imshow("Estimate Skeleton",img)
    # my_label2.configure(image=img)
    # my_label2.image = img
    print('Estimated')
    # my_label2.configure(image=img3)
    # my_label2.image = img3
    
    
def etmcam():
    pTime = 0
    detector=PoseDetector()
    cap=cv2.VideoCapture(0)
    while True:
        success,img=cap.read()
        img=detector.findPose(img)
        lmList,bboxInfo=detector.findPosition(img,bboxWithHands=True)
        cTime = time.time() # frame rate
        fps_rate = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(img, f'FPS: {int(fps_rate)}', (20, 50), cv2.FONT_HERSHEY_PLAIN,3,(0, 255, 0), 4)
        cv2.imshow("Estimate Skeleton",img)
    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            cap.release() 
            break
  
def etmcamcnn():
    pTime = 0
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', help=r'C:\Users\Admin\Downloads\Human-Pose-Estimation-master\human-pose-estimation-opencv-master')
    parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
    parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
    parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')
    args = parser.parse_args()

    BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

    POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

    inWidth = args.width
    inHeight = args.height
    net = cv2.dnn.readNetFromTensorflow(r"C:\Users\Admin\Downloads\graph_opt.pb")
    cap=cv2.VideoCapture(0)
    while True:
        hasFrame, frame = cap.read()
        if not hasFrame:
            cv2.waitKey()
            break
        frameWidth = frame.shape[1]
        frameHeight = frame.shape[0]
    
        net.setInput(cv2.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
        out = net.forward()
        out = out[:, :19, :, :]  # MobileNet output [1, 57, -1, -1], we only need the first 19 elements

        assert(len(BODY_PARTS) == out.shape[1])

        points = []
        for i in range(len(BODY_PARTS)):
            # Slice heatmap of corresponging body's part.
            heatMap = out[0, i, :, :]

            # Originally, we try to find all the local maximums. To simplify a sample
            # we just find a global one. However only a single pose at the same time
            # could be detected this way.
            _, conf, _, point = cv2.minMaxLoc(heatMap)
            x = (frameWidth * point[0]) / out.shape[3]
            y = (frameHeight * point[1]) / out.shape[2]
            # Add a point if it's confidence is higher than threshold.
            points.append((int(x), int(y)) if conf > args.thr else None)
        for pair in POSE_PAIRS:
            partFrom = pair[0]
            partTo = pair[1]
            assert(partFrom in BODY_PARTS)
            assert(partTo in BODY_PARTS)

            idFrom = BODY_PARTS[partFrom]
            idTo = BODY_PARTS[partTo]

            if points[idFrom] and points[idTo]:
                cv2.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
                cv2.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
                cv2.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv2.FILLED)
            
        cTime = time.time() # frame rate
        fps_rate = 1 / (cTime-pTime)
        pTime = cTime
        cv2.putText(frame, f'FPS: {int(fps_rate)}', (20, 50), cv2.FONT_HERSHEY_PLAIN,3,(0, 255, 0), 4)
        cv2.imshow("Estimate Skeleton",frame)
    
        k = cv2.waitKey(30) & 0xff
        if k == 27:
            cv2.destroyAllWindows()
            cap.release() 
            break    


    #%bt image
button_image = Button(frame2, text='Estimate skeleton a image', command=estmimg,bg='#008E89', fg='white',width=27)
button_image.pack(ipadx=15,ipady=15)
  #%btestimate skeleton cnn
button_estm = Button(frame2, text='Estimate skeleton cam real time CNN', command=etmcamcnn,bg='#C1F4C5',width=27)
button_estm.pack(pady=5,ipadx=15,ipady=15)
    #%btestimate skeleton
button_estm = Button(frame2, text='Estimate skeleton cam real time', command=etmcam,bg='#C1F4C5',width=27)
button_estm.pack(pady=5,ipadx=15,ipady=15)  


    #%btquit
button_quit = Button(frame2, text='Exit Program', command=root.destroy,bg='#EF6D6D',width=27)
button_quit.pack(pady=5,ipadx=15,ipady=15)

    #%btn-facedetection
button_estmcam = Button(frame3, text='Facedetection- in real time', command=facedect,bg='#008E89',fg='white',width=27)
button_estmcam.pack(pady=5,ipadx=15,ipady=15)   
    #%btn -handtracking
button_estmcam = Button(frame3, text='Hand tracking in real time', command=handtrk,bg='#C1F4C5',width=27)
button_estmcam.pack(pady=5,ipadx=15,ipady=15)   
    #%btn - Face Mesh detection
button_estmcam = Button(frame3, text='Face Mesh detection in real time', command=facemsh,bg='#FFBED8',width=27)
button_estmcam.pack(pady=5,ipadx=15,ipady=15)
    #%btn combination estimate skeleton
button_estmcam = Button(frame3, text='Combination Estimate Cam in real time', command=combinationetm,bg='#FFBED8',width=27)
button_estmcam.pack(pady=5,ipadx=15,ipady=15)   

root.mainloop()