from re import L
import cv2
import numpy as np
import face_recognition as face_rec



def rebulid(img,size) :
    width = int(img.shape[1]*size)
    height = int(img.shape[0]*size)
    dimension  = (width,height)
    return cv2.rebulid(img,dimension,interpolation=cv2.INTER_AREA)

train_path = './train_images/train.jpg'
test_path = './test_images/test.jpg'
# Loading the image and resizing it.
filename = face_rec.load_image_file(train_path)
filename = cv2.cvtColor(filename,cv2.COLOR_BGR2RGB)
filename = rebulid(filename,0.100)
filename_test = face_rec.load_image_file(test_path)
filename_test = cv2.cvtColor(filename_test,cv2.COLOR_BGR2RGB)
filename_test = rebulid(filename_test,0.100)



# Finding the face location and encoding the face.
faceLocation_filename = face_rec.face_locations(filename)[0]
encode_filename = face_rec.face_encodings(filename)[0]
cv2.rectangle(filename, (faceLocation_filename[3], faceLocation_filename[0]), (faceLocation_filename[1], faceLocation_filename[2]), (255, 0, 255), 3)


faceLocation_filename_test = face_rec.face_locations(filename_test)[0]
encode_filename_test = face_rec.face_encodings(filename_test)[0]
cv2.rectangle(filename_test, (faceLocation_filename[3], faceLocation_filename[0]), (faceLocation_filename[1], faceLocation_filename[2]), (255, 0, 255), 3)

results = face_rec.compare_faces([encode_filename], encode_filename_test)
print(results)
cv2.putText(filename_test, f'{results}', (50,50), cv2.FONT_HERSHEY_COMPLEX, 1,(0,0,255), 2 )

cv2.imshow('main', filename)
cv2.imshow('test', filename_test)
cv2.waitKey(0)
cv2.destroyAllWindows()