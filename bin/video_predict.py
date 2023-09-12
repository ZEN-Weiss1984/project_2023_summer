import model_train
from model_train import Net
import sys
import cv2
import os
from sys import platform
import argparse
import math
import torch
sys.path.append('E:/Compiler_source/openpose/openpose1.7.0/build/bin')
os.environ['PATH']  = os.environ['PATH'] + ';' + 'E:/Compiler_source/openpose/openpose1.7.0/build/bin;' + 'E:/Compiler_source/openpose/openpose1.7.0/build/bin;'
import pyopenpose as op

def loc_calcu(base_point,input_point,len):
    output_point=[]
    output_point.append((input_point[0]-base_point[0])/len)
    output_point.append((input_point[1]-base_point[1])/len)
    output_point.append(input_point[2])
    return output_point

params = dict()
params["model_folder"] = "e:/Compiler_source/openpose/openpose1.7.0/models"
params["face"] = False
params["hand"] = False
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

datum = op.Datum()
#cap = cv2.VideoCapture(r"C:\Users\k\Desktop\body_pre\data_in\test_long\test2.mp4")
#cap = cv2.VideoCapture(r"C:\Users\k\Pictures\Camera Roll\WIN_20230902_13_12_44_Pro.mp4")
cap=cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FPS, 32)

model_pth="C:/Users/k/Desktop/body_pre/models/"
model_pth=model_pth+"lstm3"+".pth"
model=torch.load(model_pth)
predict=0
iter=0
hidden_prev=torch.zeros(model_train.NUM_LAYERS,1,model_train.HIDDEN_SIZE).cuda()
c_hint = torch.zeros(model_train.NUM_LAYERS,1, model_train.HIDDEN_SIZE).cuda()

while True:
    # 读取摄像头帧
    ret, frame = cap.read()

    if not ret:
        break

    imageToProcess = frame
    datum.cvInputData = imageToProcess
    opWrapper.emplaceAndPop(op.VectorDatum([datum]))
    frame=datum.cvOutputData
    cv2.putText(frame,str(cap.get(cv2.CAP_PROP_FPS)),(400, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)
    if iter%4==0:
        if (datum.poseKeypoints[0].tolist())[1][2] >= 0.25:
            base_point=datum.poseKeypoints[0].tolist()
            base_point=[base_point[1][0],base_point[1][1]]

            se_point=datum.poseKeypoints[0].tolist()
            se_point=[se_point[2][0],se_point[2][1]]

            base_len=math.sqrt(math.pow(base_point[0]-se_point[0],2)+math.pow(base_point[1]-se_point[1],2))
            point_list=loc_calcu(base_point,(datum.poseKeypoints[0][2]).tolist(),base_len)+loc_calcu(base_point,(datum.poseKeypoints[0][3]).tolist(),base_len)+loc_calcu(base_point,(datum.poseKeypoints[0][4]).tolist(),base_len)
            point_list=torch.tensor(point_list).cuda()

            out,accuracy,hidden_prev,c_hint=model_train.model_pre(model,point_list,hidden_prev,c_hint)
            predict=out
            print(f"out : {out}, accuracy : {accuracy}")
        else:
            print("human not found")

    iter+=1
    cv2.putText(frame,str(predict),(200, 100), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)


    # 显示结果
    cv2.imshow("OpenPose", frame)
    # 按q退出
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭OpenPose
cap.release()