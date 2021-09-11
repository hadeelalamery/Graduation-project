import os
import cv2 as cv
from sklearn.model_selection import train_test_split, KFold
from keras.layers import Dense,Activation,Conv2D, MaxPooling2D, Flatten, Dropout, Softmax, BatchNormalization, Conv1D, MaxPool1D, MaxPool2D

# Import Keras with tensorflow backend
from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing import image
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential, Model
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, Convolution2D
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications import vgg16
from keras.callbacks import EarlyStopping, ModelCheckpoint, TensorBoard
from matplotlib import pyplot as plt
# import miscellaneous modules
import cv2

from keras.utils import to_categorical
# you can change directory to your own path
directory = "/Users/UserName/Desktop/Iris Recognition/CASIA Iris Image Database (version 1.0)"


import numpy as np
# read all the images from the directory
classes = 108
#read image from dataset
def readimage(directory, dataset):
    #read image from dataset
    #vn = 0
    files = []
    for first_sub in os.listdir(directory):
        #if vn == 0:
           
          for second_sub in os.listdir(directory+"/"+first_sub+"/{}".format(dataset) ):
            #vn+=1
            if second_sub[-3:] == 'bmp':
                files.append([directory+"/"+first_sub+"/{}".format(dataset) +"/"+second_sub ,first_sub])
    return files
file_name_list = []
images = []
data = []
labels = []
ile_name_list = readimage(directory, 1) + readimage(directory, 2)


images = []
data = []
labels = []
save_folder = 'CASIA-v1 Dataset_sementation_cropping_entire_iris'
if not os.path.isdir(save_folder):
       os.makedirs(save_folder)


contt = 0
for file_name in file_name_list:

    #You can change the number 399

    img = cv.imread('%s' %  file_name[0],0)
    img22 = img.copy()
    lable = int(file_name[1])

    xp = np.argmin(np.sum(img, axis=0)[100:180]) + 100
    yp = np.argmin(np.sum(img, axis=1)[100:180]) + 100
    # img = img[yp - 80:yp + 80, xp - 100:xp + 100]
    for i in range(2):
        region = img[yp - 60: yp + 60, xp - 60: xp + 60]
        retval, dst = cv.threshold(region, 65, 1, cv.THRESH_BINARY)
        mask = np.where(dst != 0, 1, 0)
        xp += np.argmin(np.sum(mask, axis=0)) - 60
        yp += np.argmin(np.sum(mask, axis=1)) - 60

    width1 = 110
    region_inner = img[max(0, yp - width1):min(280, yp + width1),
                   max(0, xp - width1):min(320, xp + width1)]
    width2 = 125 
    region_outer = img[max(0, yp - width2):min(280, yp + width2),
                   max(0, xp - width2):min(320, xp + width2)]
    inner_filter = cv.bilateralFilter(region_inner, 9, 75, 75)
    inner_filter = cv.medianBlur(inner_filter, 9)
    # inner_filter = cv.GaussianBlur(inner_filter, (15, 15), 150)
    var = 0.33
    median = np.median(region_inner)
    para1 = int(max(0, (1.0 - var) * median))
    para2 = int(min(255, (1.0 + var) * median))
    height,width = img.shape
    mask = np.zeros((height,width), np.uint8)

    # use canny edge detector to get an image of inner boundary
    edged_inner = cv.Canny(inner_filter, para1, para2)
    inner_circles = cv.HoughCircles(edged_inner, cv.HOUGH_GRADIENT, 1, 300,
                                    param1=50, param2=10, minRadius=25, maxRadius=48)
    # region_outer = cv.GaussianBlur(region_outer, (15, 15), 150)
    region_outer = cv.bilateralFilter(region_outer, 9, 95, 95)
    outer_circles = cv.HoughCircles(region_outer, cv.HOUGH_GRADIENT, 1, 300, param1=30, param2=10, minRadius=95,
                                    maxRadius=114)
   
    # draw circles
    # pupil boundary
    for i in inner_circles[0, :]:
        inner_circle = [int(i[0]) + xp - width1, int(i[1]) + yp - width1, i[2]]
        cv.circle(img, (inner_circle[0], inner_circle[1]), inner_circle[2], color=(255, 0, 0), thickness=1)
    # cv.imshow('segme', img)

    
    # iris boundary
    for i in outer_circles[0, :]:
        outer_circle = [int(i[0]) + xp - width2, int(i[1]) + yp - width2, i[2]]
    inner_x, inner_y, inner_r = map(int, map(int, inner_circle))
    outer_x, outer_y, outer_r = map(int, map(int, outer_circle))
    area_1_x = max(0, inner_x - inner_r - 64)
    area_2_x = inner_x + inner_r
    area_y = inner_y - 30
    area_1_h = 60
    area_2_h = 60
    # img_bgr = cv.cvtColor(img, cv.COLOR_GRAY2BGR)
    img_bgr = img
    cv.circle(img_bgr, (inner_x, inner_y), inner_r, (255, 0, 0), 2)
    cv.circle(img_bgr, (outer_x, outer_y), (outer_r + 45), (255, 0, 0),2)
    cv2.circle(mask,(outer_x, outer_y),outer_r,(255,255,255),thickness=-1)
    #-------------------------------------------------#

    masked_data = cv2.bitwise_and(img22, img22, mask=mask)
     # Apply Threshold
    _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

    # Find Contour
    _, contours, _ = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
    x,y,w,h = cv2.boundingRect(contours[0])

    # Crop masked_data
    crop500 = masked_data[y:y+h,x:x+w]


    #cv.imshow('marege', marege)
    # both_irises = marege
    both_sides = cv.cvtColor(crop500, cv.COLOR_GRAY2BGR)
    both_sides = cv.resize(both_sides, (400, 400))
    image = both_sides
    image = np.array(image)
    data.append(image)
    labels.append(lable)
    new_path = os.path.join(save_folder,str(lable))
    if not os.path.isdir(new_path):
       os.makedirs(new_path)
    
    cv2.imwrite("{}".format(os.path.join(new_path ,'{}.bmp'.format(contt))) ,image)
    contt+=1
    


#Converting lists into numpy arrays
data = np.array(data)
labels = np.array(labels)

print("data.shape is", data.shape, "labels.shape is", labels.shape)


x_train, x_test, y_train, y_test = train_test_split(data, labels, test_size=0.1, random_state=42)
print(x_train.shape, x_test.shape, y_train.shape, y_test.shape)
y_train = to_categorical(y_train, 108)
y_test = to_categorical(y_test, 108)

# Create the model
# X_train = X_train/225.0
# X_train =  X_train.reshape(X_train[0], )
print("x_train.shape", x_train.shape, "x_test.shape", x_test.shape, "y_train.shape", y_train.shape, "y_test.shape",
      y_test.shape)


def create_model():

    LEARNING_RATE = 0.0001 
    
    #X_train = X_train/225.0
    #X_train =  X_train.reshape(X_train[0], )

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
    classifier.add(Dense(108, activation='softmax'))
    # sigmoid helps in 0 1 classification
    
    # Compiling the CNN
    classifier.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
    
    # Deffining the Training and Testing Datasets
    
    
    
    classifier.summary()
    early_stopper = EarlyStopping(monitor='val_acc', min_delta=0.001, patience=30, verbose=1)
    #tensorboard = TensorBoard(log_dir='logs/', write_graph=False)
    
    
    # nb_epochs how much times you want to back propogate
    # steps_per_epoch it will transfer that many images at 1 time
    # & epochs means 'steps_per_epoch' will repeat that many times
    history = classifier.fit( x_train, y_train, epochs=40, batch_size=4, validation_data=(x_test, y_test), callbacks=[early_stopper])

    classifier.save('model2_save.h5')
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

