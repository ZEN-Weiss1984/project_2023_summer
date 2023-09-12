import sys
import cv2
import os
from sys import platform
import argparse
import json
import math
sys.path.append('E:/Compiler_source/openpose/openpose1.7.0/build/bin')
os.environ['PATH']  = os.environ['PATH'] + ';' + 'E:/Compiler_source/openpose/openpose1.7.0/build/bin;' + 'E:/Compiler_source/openpose/openpose1.7.0/build/bin;'
import pyopenpose as op

params = dict()
params["model_folder"] = "e:/Compiler_source/openpose/openpose1.7.0/models"
params["face"] = False
params["hand"] = False
opWrapper = op.WrapperPython()
opWrapper.configure(params)
opWrapper.start()

train_set=[]
train_target=[]

def loc_calcu(base_point,input_point,len):
    output_point=[]
    output_point.append((input_point[0]-base_point[0])/len)
    output_point.append((input_point[1]-base_point[1])/len)
    output_point.append(input_point[2])
    return output_point


def getdata(file_root_path,datalen):
    global train_set,train_target
    print(f"data in {file_root_path}")
    for i in range(datalen):
        file_path=file_root_path+"picture/"+str(i+1)+"/"

        pic_root_path=file_path+"frame_"
        point_list=[]
        for j in range(64):
            pic_path=pic_root_path+str('%04d' % (j+1))+".jpg"

            datum = op.Datum()
            imageToProcess = cv2.imread(pic_path,cv2.IMREAD_COLOR)
            #print(pic_path)
            datum.cvInputData = imageToProcess
            opWrapper.emplaceAndPop(op.VectorDatum([datum]))

            base_point=datum.poseKeypoints[0].tolist()
            base_point=[base_point[1][0],base_point[1][1]]

            se_point=datum.poseKeypoints[0].tolist()
            se_point=[se_point[2][0],se_point[2][1]]

            base_len=math.sqrt(math.pow(base_point[0]-se_point[0],2)+math.pow(base_point[1]-se_point[1],2))
            point_list.append(
                loc_calcu(base_point,(datum.poseKeypoints[0][2]).tolist(),base_len)+loc_calcu(base_point,(datum.poseKeypoints[0][3]).tolist(),base_len)+loc_calcu(base_point,(datum.poseKeypoints[0][4]).tolist(),base_len)
                )
        train_set.append(point_list)


        target_path=file_path+"data.json"
        f=open(target_path,"r")
        content=f.read()
        dic=json.loads(content)
        target_list=[]
        for j in range(64):
            t=dic[j]['class']
            target_piece=[]
            for p in range(4):
                if p==t:
                    target_piece.append(1)
                else:
                    target_piece.append(0)
            target_list.append(target_piece)
        f.close()
        train_target.append(target_list)

        print(f"  finished : {i+1}")



getdata(file_root_path="C:/Users/k/Desktop/body_pre/data_in/test3/",datalen=60)
getdata(file_root_path="C:/Users/k/Desktop/body_pre/data_in/test4/",datalen=45)






train_set_json=json.dumps(train_set)
train_target_json=json.dumps(train_target)

fs=open(r"C:/Users/k/Desktop/body_pre/data_detected/train_set.json","w")
fs.write(train_set_json)
fs.close()
fs=open(r"C:/Users/k/Desktop/body_pre/data_detected/train_target.json","w")
fs.write(train_target_json)
fs.close()

print("data saved!")