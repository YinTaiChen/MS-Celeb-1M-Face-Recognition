import dlib
import numpy
from skimage import io
import cv2
import os

def get_landmarks(im):

    rects = detector(im, 1)

    return numpy.matrix([[p.x, p.y] for p in predictor(im, rects[0]).parts()])

predictor_path = "./data/shape_predictor_68_face_landmarks.dat"
detector = dlib.get_frontal_face_detector()
predictor = dlib.shape_predictor(predictor_path)

with open("MS-Celeb-1M_clean_list.txt", "r") as data_file:

    sliced_data = open("sliced_data.txt", 'a')

    for line in data_file.readlines():
        faces_path = "./msr/"+line.split()[0]
        face = cv2.imread(faces_path)
        label = line.split()[1]

        try:
            img = io.imread(faces_path)
            dets = detector(img, 1)
            # print("Number of faces detected: {}".format(len(dets)))
        except Exception as e:
            print(str(e))

        if len(dets) == 1:
            landmarks = get_landmarks(img)
            landmark_list = []

            for k in range(int(landmarks.size/2)):
                if len(str(landmarks[k])) == 11:
                    landmark_list.append([int(str(landmarks[k])[2:5]), int(str(landmarks[k])[6:9])])
                else:
                    landmark_list.append([int(str(landmarks[k])[2:4]), int(str(landmarks[k])[5:7])])
            for l in range(len(landmark_list)):
                x = landmark_list[l][0] - 10
                y = landmark_list[l][1] - 10
                sliced = face[x:x+20, y:y+20]
                prefix = "/media/dannychen/TOURO Mobile"
                path = prefix+"/sliced_2/"+line.split()[0].split("/")[0]+"/"+str(l)+"_"+line.split()[0].split("/")[1]
                directory = prefix+"/sliced_2/"+line.split()[0].split("/")[0]

                if not os.path.exists(directory):
                    os.makedirs(directory)

                cv2.imwrite(path, sliced)
                sliced_data.write(line.split()[0].split("/")[0]+"/"+str(l)+"_"+line.split()[0].split("/")[1]+" "+str(label)+"\n")
