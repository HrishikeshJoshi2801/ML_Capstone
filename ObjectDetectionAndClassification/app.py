import tkinter as tk
from tkinter import filedialog
import os
from PIL import ImageTk, Image

def start():
    root = tk.Tk()
    root.title("Object Detector!")

    canvas = tk.Canvas(root, height=600, width=700, bg="#263D42")
    canvas.pack()

    frame = tk.Frame(root, bg="white")
    frame.place(relwidth=0.8, relheight=0.8, relx=0.1, rely=0.1)

    def openImg():
        filename = filedialog.askopenfilename(initialdir="/", title="Select File",
                                          filetype=(("image", "*.jpg"), ("image", "*.jpeg"), ("image", "*.png"), ("image", "*.gif"), ("all files", "*.*")))
        # print(filename)
        img = cv2.imread(filename)
        cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
        
        ClassIndex , confidece , bbox = model.detect(img,confThreshold=0.6)

        font_scale = 3
        font = cv2.FONT_HERSHEY_PLAIN

        for classInd , conf ,boxes in zip(ClassIndex.flatten(),confidece.flatten(),bbox):
            cv2.rectangle(img,boxes ,(255,0,0),2)
            cv2.putText(img,classLabels[classInd-1],(boxes[0]+10,boxes[1]+40),font , fontScale = font_scale , color =(0,255,0),thickness=3)
        
        plt.imshow(cv2.cvtColor(img,cv2.COLOR_BGR2RGB))
        plt.show()
        print("Done")
            
    def openVDO():
        filename = filedialog.askopenfilename(initialdir="/", title="Select File",
                                          filetype=(("video", "*.mp4"), ("video", "*.mkv"), ("video", "*.avi"), ("video", "*.mov"), ("all files", "*.*")))

        cap = cv2.VideoCapture(filename)

        #check if the video is opened correctly

        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise IOError("Cannot open video")

        font_scale = 3
        font = cv2.FONT_HERSHEY_PLAIN

        while True:
            ret, frame = cap.read()
        
            ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.6)

            print(ClassIndex)
            

            if (len(ClassIndex) != 0):
                for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
                    if (ClassInd <= 80):
                        cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                        cv2.putText(frame, classLabels[ClassInd-1], (boxes[0] + 10,boxes[1]+40), font, fontScale = font_scale, color=(0, 225, 0))
        
            cv2.imshow('Object Detection Tutorial', frame)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllwindows()

    def openWebCam():
        cap = cv2.VideoCapture(0)

        #check if the video is opened correctly

        if not cap.isOpened():
            cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            raise IOError("Cannot open WebCam")

        font_scale = 3
        font = cv2.FONT_HERSHEY_PLAIN

        while True:
            ret, frame = cap.read()
        
            ClassIndex, confidece, bbox = model.detect(frame, confThreshold=0.6)

            print(ClassIndex)
            

            if (len(ClassIndex) != 0):
                for ClassInd, conf, boxes in zip(ClassIndex.flatten(), confidece.flatten(), bbox):
                    if (ClassInd <= 80):
                        cv2.rectangle(frame, boxes, (255, 0, 0), 2)
                        cv2.putText(frame, classLabels[ClassInd-1], (boxes[0] + 10,boxes[1]+40), font, fontScale = font_scale, color=(0, 225, 0))
        
            cv2.imshow('Object Detection Tutorial', frame)

            if cv2.waitKey(2) & 0xFF == ord('q'):
                break

        cap.release()
        cv2.destroyAllwindows()


    openImg = tk.Button(frame, text="Open Image", padx=10, pady=5, fg="white", bg="#263D42", command=openImg)
    openImg.place(relwidth=0.2, relheight=0.15, relx=0.1, rely=0.5)

    openVDO = tk.Button(frame, text="Open Video", padx=10, pady=5, fg="white", bg="#263D42", command=openVDO)
    openVDO.place(relwidth=0.2, relheight=0.15, relx=0.4, rely=0.5)

    openWebCam = tk.Button(frame, text="Open Web-cam", padx=10, pady=5, fg="white", bg="#263D42", command=openWebCam)
    openWebCam.place(relwidth=0.2, relheight=0.15, relx=0.7, rely=0.5)

    root.mainloop()

if __name__ == "__main__":
    import cv2
    import matplotlib.pyplot as plt

    config_file = 'ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    frozen_model = 'frozen_inference_graph.pb'

    model = cv2.dnn_DetectionModel(frozen_model,config_file)

    classLabels =[]
    file_name = 'Labels.txt'
    with open(file_name ,mode='rt')as fpt:
        classLabels = fpt.read().rstrip('\n').split('\n')

    model.setInputSize(320,320)
    model.setInputScale(1.0/127.5)  #255/2 = 127.5
    model.setInputMean((127.5 ,127.5 ,127.5)) #mobilenet =>[-1 , 1]
    model.setInputSwapRB(True)

    start()