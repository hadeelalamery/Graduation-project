import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
import tensorflow as tf
from PIL import Image
import os
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical
from keras.models import Sequential, load_model
from keras.layers import Conv2D, MaxPool2D, Dense, Flatten, Dropout
import cv2 as cv

import cv2
from os import listdir,mkdir
from os.path import isdir, join, isfile, splitext
import os
import re

directory = "Dataset"
images = []
data = []
labels = []
classes = 50  #number trafic sgin
cur_path = os.getcwd()
file_name = []




train_dir = 'ATVS_Dataset/'


save_folder = 'ATVS_segmentation_Entire_Iris'
if not os.path.isdir(save_folder):
       os.makedirs(save_folder)



def get_image_files_only(folder):
    return [os.path.join(folder, f) for f in os.listdir(folder) if re.match(r'.*\.(jpg|jpeg|png|bmp|tif)', f, flags=re.I)]



num_name = 0

obj = 0
for class_dir in listdir(train_dir):
    if not isdir(join(train_dir, class_dir)):
        continue
    Name = class_dir


    mm = 0
    for img_path in get_image_files_only(join(train_dir, class_dir)):

           img = cv.imread(img_path,0)
           #img = cv2.resize(image, (200, 200))
           img22 = img.copy()
           xp = np.argmin(np.sum(img, axis=0)[200:240]) + 200
           yp = np.argmin(np.sum(img, axis=1)[200:240]) + 200
           for i in range(2):
               region = img[yp - 70: yp + 70, xp - 70: xp +
                                            70]
               retval, dst = cv.threshold(region, 65, 1, cv.THRESH_BINARY)
               mask = np.where(dst != 0, 1, 0)

               xp += np.argmin(np.sum(mask, axis=0)) - 60
               yp += np.argmin(np.sum(mask, axis=1)) - 60
           width1 =110
           region_inner = img[max(0, yp - width1):min(
               480, yp + width1),
                          max(0, xp - width1):min(640, xp + width1)]
           width2 = 125
           region_outer = img[max(0, yp - width2):min(480, yp + width2),
                          max(0, xp - width2):min(640, xp + width2)]
           inner_filter = cv.bilateralFilter(region_inner, 9, 75, 75)
           var = 0.33
           median = np.median(region_inner)
           para1 = int(max(0, (1.0 - var) * median))
           para2 = int(min(255, (1.0 + var) * median))
           height, width = img.shape
           mask = np.zeros((height, width), np.uint8)

           # use canny edge detector to get an image of inner boundary
           edged_inner = cv.Canny(inner_filter, para1, para2)
           inner_circles = cv.HoughCircles(edged_inner, cv.HOUGH_GRADIENT, 1, 600, param1=50, param2=10, minRadius=25,
                                           maxRadius=68)  # change max to 68 to increase y rud

           region_outer = cv.bilateralFilter(region_outer, 9, 95, 95)
           outer_circles = cv.HoughCircles(region_outer, cv.HOUGH_GRADIENT, 1, 600, param1=30, param2=10, minRadius=95,
                                           maxRadius=114)

           # draw circles
           # pupil boundary
           for i in inner_circles[0, :]:
               inner_circle = [int(i[0]) + xp - width1, int(i[1]) + yp - width1, i[2]]
               cv.circle(img, (inner_circle[0], inner_circle[1]), inner_circle[2], color=(255, 0, 0), thickness=1)
           # cv.imshow('h', img)
           # cv.imwrite('atvs_pupil*.bmp', img)
           # iris boundary

           for i in outer_circles[0, :]:
               outer_circle = [int(i[0]) + xp - width2, int(i[1]) + yp - width2, i[2]]
               # cv.circle(img, (outer_circle[0], outer_circle[1]), outer_circle[2], color=(255, 0, 0), thickness=2)
               # cv.imshow('seg', img)
               # cv.imwrite('atvs_iris.bmp', img)

               inner_x, inner_y, inner_r = map(int, map(int, inner_circle))
               outer_x, outer_y, outer_r = map(int, map(int, outer_circle))
               area_1_x = max(0, inner_x - inner_r - 64)
               area_2_x = inner_x + inner_r
               area_y = inner_y - 30
               area_1_h = 60
               area_2_h = 60
               # = cv.cvtColor(img, cv.COLOR_BGR2GRAY)
               img_bgr = img
               cv.circle(img_bgr, (inner_x, inner_y), inner_r, (255, 0, 0), 2)
               cv.circle(img_bgr, (outer_x, outer_y), (outer_r + 45), (255, 0, 0), 2)
               cv2.circle(mask, (outer_x, outer_y), outer_r, (255, 255, 255), thickness=-1)
               # -------------------------------------------------#

               masked_data = cv2.bitwise_and(img22, img22, mask=mask)
               # Apply Threshold
               _, thresh = cv2.threshold(mask, 1, 255, cv2.THRESH_BINARY)

               # Find Contour
               _, contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
               x, y, w, h = cv2.boundingRect(contours[0])

               # Crop masked_data
               crop500 = masked_data[y:y + h, x:x + w]

               # cv.imshow('marege', marege)
               # both_irises = marege
               both_sides = cv.cvtColor(crop500, cv.COLOR_GRAY2BGR)
               #both_sides = cv.resize(both_sides, (300, 300))
               image = both_sides
               image = cv.resize(image, (400, 400))
               image = np.array(image)
               data.append(image)
               labels.append(lable)
               #image = cv.cvtColor(image, cv.COLOR_GRAY2BGR)


               #cv.imshow('ff', image)
               #key = cv.waitKey()

               image_name = '{}.bmp'.format(Name)

               #print(image_name)
               new_path = os.path.join(save_folder,str(Name))

               if not os.path.isdir(new_path):
                 os.makedirs(new_path)

               cv2.imwrite("{}".format(os.path.join(new_path, '{}.bmp'.format(mm))), image)
               mm = mm + 1
    obj+=1


# Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print("data.shape is", data.shape, "labels.shape is", labels.shape)

x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
y_train = to_categorical(y_train, 50)
y_test = to_categorical(y_test, 50)

# Create the model
# X_train = X_train/225.0
# X_train =  X_train.reshape(X_train[0], )
print("x_train.shape", x_train.shape, "x_test.shape", x_test.shape, "y_train.shape", y_train.shape, "y_test.shape",
      y_test.shape)


def create_model():
    LEARNING_RATE = 0.0001

    # X_train = X_train/225.0
    # X_train =  X_train.reshape(X_train[0], )

    classifier = Sequential()

    # Step1 - Convolution
    # Input Layer/dimensions
    # Step-1 Convolution
    # 64 is number of output filters in the convolution
    # 3,3 is filter matrix that will multiply to input_shape=(64,64,3)
    # 64,64 is image size we provide
    # 3 is rgb
    classifier.add(Conv2D(64, strides=4, kernel_size=(5, 5), input_shape=(400, 400, 3), activation='relu'))
    classifier.add(BatchNormalization())

    # Step2 - Pooling
    # Processing
    # Hidden Layer 1
    # 2,2 matrix rotates, tilts, etc to all the images
    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # Adding a second convolution layer
    # Hidden Layer 2
    # relu turns negative images to 0
    classifier.add(Conv2D(64, strides=4, kernel_size=(5, 5), activation='relu'))

    classifier.add(MaxPooling2D(pool_size=(2, 2)))

    # step3 - Flattening
    # converts the matrix in a singe array
    classifier.add(Flatten())

    # Step4 - Full COnnection
    # 128 is the final layer of outputs & from that 1 will be considered ie dog or cat
    classifier.add(Dense(256, activation='relu'))
    classifier.add(Dense(50, activation='softmax'))
    # sigmoid helps in 0 1 classification

    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # Deffining the Training and Testing Datasets

    classifier.summary()
    early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=30, verbose=1)
    # tensorboard = TensorBoard(log_dir='logs/', write_graph=False)

    # nb_epochs how much times you want to back propogate
    # steps_per_epoch it will transfer that many images at 1 time
    # & epochs means 'steps_per_epoch' will repeat that many times
    history = classifier.fit(x_train, y_train, epochs=40, batch_size=4, validation_data=(x_test, y_test),
                             callbacks=[early_stopper])

    classifier.save('model_save.h5')
    return classifier, history, x_train, x_test, y_train, y_test


classifi, history, x_train, x_test, y_train, y_test = create_model()

y_pred_train = classifi.predict_classes(x_train)
y_pred_test = classifi.predict_classes(x_test)

# classes = [ 'user_1',


report_train = classification_report(np.argmax(y_train, axis=1), y_pred_train)

with open('classification_report_train.txt', 'w') as f:
    f.write("%s\n" % report_train)

print(report_train)

print("###########################################_TEST_#########################################################")

report_test = classification_report(np.argmax(y_test, axis=1), y_pred_test)

with open('classification_report_test.txt', 'w') as f:
    f.write("%s\n" % report_test)

print(report_test)

classifier, history = create_model()

plt.plot(history.history['acc'])
plt.plot(history.history['loss'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['accuracy', 'loss'], loc='upper left')
plt.show()

plt.plot(history.history['val_acc'])
plt.plot(history.history['val_loss'])
plt.title('model val accuracy')
plt.ylabel('val_accuracy')
plt.xlabel('epoch')
plt.legend(['val_accuracy', 'val_loss'], loc='upper left')
plt.show()




