# Face Gender and Age Detection using Computer Vision and CaffeNet CNN

import cv2
import math
import argparse
import time

def highlightFace(net, frame, conf_threshold=0.7):
    frameOpencvDnn = frame.copy()
    frameHeight = frameOpencvDnn.shape[0]
    frameWidth = frameOpencvDnn.shape[1]
    blob = cv2.dnn.blobFromImage(frameOpencvDnn, 1.0, (300, 300), [104, 117, 123], True, False)

    net.setInput(blob)
    start = time.time()
    detections = net.forward()
    face_detection_time = time.time() - start
    faceBoxes = []
    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > conf_threshold:
            x1 = int(detections[0,0,i,3]*frameWidth)
            y1 = int(detections[0,0,i,4]*frameHeight)
            x2 = int(detections[0,0,i,5]*frameWidth)
            y2 = int(detections[0,0,i,6]*frameHeight)
            faceBoxes.append([x1,y1,x2,y2])
            cv2.rectangle(frameOpencvDnn, (x1,y1), (x2,y2), (0,255,0), int(round(frameHeight/150)), 8)
    return frameOpencvDnn, faceBoxes, face_detection_time, confidence

parser = argparse.ArgumentParser()
parser.add_argument('--image')

args = parser.parse_args()

faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male','Female']

faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

video = cv2.VideoCapture(args.image if args.image else 0)
padding = 20

while cv2.waitKey(1) < 0:
    hasFrame, frame = video.read()
    if not hasFrame:
        cv2.waitKey()
        break
    
    start_time = time.time()
    resultImg, faceBoxes, face_detection_time, face_confidence = highlightFace(faceNet, frame)
    if not faceBoxes:
        print("No face detected")
        continue

    for faceBox in faceBoxes:
        face = frame[max(0,faceBox[1]-padding):
                   min(faceBox[3]+padding,frame.shape[0]-1),max(0,faceBox[0]-padding)
                   :min(faceBox[2]+padding, frame.shape[1]-1)]

        blob = cv2.dnn.blobFromImage(face, 1.0, (227,227), MODEL_MEAN_VALUES, swapRB=False)
        
        genderNet.setInput(blob)
        gender_start = time.time()
        genderPreds = genderNet.forward()
        gender_time = time.time() - gender_start
        gender = genderList[genderPreds[0].argmax()]
        gender_confidence = genderPreds[0].max() * 100

        ageNet.setInput(blob)
        age_start = time.time()
        agePreds = ageNet.forward()
        age_time = time.time() - age_start
        age = ageList[agePreds[0].argmax()]
        age_confidence = agePreds[0].max() * 100

        total_time = time.time() - start_time

        print(f'Gender: {gender} (Confidence: {gender_confidence:.2f}%)')
        print(f'Age: {age[1:-1]} years (Confidence: {age_confidence:.2f}%)')
        print(f'Face Detection Time: {face_detection_time:.4f} seconds')
        print(f'Gender Prediction Time: {gender_time:.4f} seconds')
        print(f'Age Prediction Time: {age_time:.4f} seconds')
        print(f'Total Processing Time: {total_time:.4f} seconds')
        print(f'Face Detection Confidence: {face_confidence:.2f}')
        print('---')

        cv2.putText(resultImg, f'{gender}, {age}', (faceBox[0], faceBox[1]-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,255), 2, cv2.LINE_AA)
        # cv2.putText(resultImg, f'Face Conf: {face_confidence:.2f}', (faceBox[0], faceBox[1]-40), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'Gender Conf: {gender_confidence:.2f}%', (faceBox[0], faceBox[1]-70), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'Age Conf: {age_confidence:.2f}%', (faceBox[0], faceBox[1]-100), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
        cv2.putText(resultImg, f'Time: {total_time:.4f}s', (faceBox[0], faceBox[1]-130), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,0), 2, cv2.LINE_AA)
    
    cv2.imshow("Detecting age and gender", resultImg)

cv2.destroyAllWindows()
