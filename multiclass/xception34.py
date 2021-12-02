#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# import pertinent libraries
import os
import sys
import datetime
import glob as glob
import numpy as np
import cv2
from collections import Counter
# [Keras Models]
# import the Keras implementations of VGG16, VGG19, InceptionV3 and Xception models
# the model used here is VGG16
from keras.applications.vgg16 import VGG16, preprocess_input
from keras.applications import InceptionV3, ResNet50, Xception
from keras.models import Model
from keras.layers import Dense, GlobalAveragePooling2D, Dropout, Flatten
from keras.preprocessing.image import ImageDataGenerator, array_to_img, img_to_array, load_img
from keras.optimizers import SGD
from keras.callbacks import ModelCheckpoint, Callback
from keras import layers, models, optimizers
import tensorflow.keras.backend as K
import itertools
from itertools import tee
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, precision_score, recall_score

#from scipy.interpolate import spline
import pandas as pd
import matplotlib.pyplot as plt
#get_ipython().run_line_magic('matplotlib', 'inline')
from sklearn.utils import class_weight

# Definition for calculating class weights
def calculating_class_weights(y_true):
    from sklearn.utils.class_weight import compute_class_weight
    number_dim = np.shape(y_true)[1]
    weights = np.empty([number_dim, 2])
    for i in range(number_dim):
        weights[i] = compute_class_weight('balanced', [0.,1.], y_true[:, i])
    return weights

# weighted binary cross entropy definition
def get_weighted_loss(weights):
    def weighted_loss(y_true, y_pred):
        return K.mean((weights[:,0]**(1-y_true))*(weights[:,1]**(y_true))*K.binary_crossentropy(y_true, y_pred), axis=-1)
    return weighted_loss

# To make sure you can see the f1, precision and recall scores of the validation set during training.
class Metrics(Callback):
    def __init__(self, validation_generator, validation_steps, threshold=0.5):
        self.validation_generator = validation_generator
        self.validation_steps = validation_steps or len(validation_generator)
        self.threshold = threshold

    def on_train_begin(self, logs={}):
        self.val_f1_scores = []
        self.val_recalls = []
        self.val_precisions = []

    def on_epoch_end(self, epoch, logs={}):
        # duplicate generator to make sure y_true and y_pred are calculated from the same observations
        gen_1, gen_2 = tee(self.validation_generator)
        y_true = np.vstack(next(gen_1)[1] for _ in range(self.validation_steps)).astype('int')
        y_pred = (self.model.predict_generator(gen_2, steps=self.validation_steps) > self.threshold).astype('int')
        f1 = f1_score(y_true, y_pred, average='macro')
        precision = precision_score(y_true, y_pred, average='macro')
        recall = recall_score(y_true, y_pred, average='macro')
        self.val_f1_scores.append(f1)
        self.val_recalls.append(recall)
        self.val_precisions.append(precision)
        print(f" - val_f1_score: {f1:.5f} - val_precision: {precision:.5f} - val_recall: {recall:.5f}")
        return 

def getListOfFiles(dirName):
    # create a list of file and sub directories 
    # names in the given directory 
    listOfFile = os.listdir(dirName)
    allFiles = list()
    # Iterate over all the entries
    for entry in listOfFile:
        # Create full path
        fullPath = os.path.join(dirName, entry)
        # If entry is a directory then get the list of files in this directory 
        if os.path.isdir(fullPath):
            allFiles = allFiles + getListOfFiles(fullPath)
        else:
            allFiles.append(fullPath)
                
    return allFiles

# [Dataset]
# image dimensions for VGG16, VGG19 are 224, 224
# image dimensions for InceptionV3 and Xception are 299, 299
img_width, img_height = 224,224

train_dir = 'Training'
validate_dir = 'Validation'
nb_epochs = 100
batch_size = 64
nb_classes = len(glob.glob(train_dir + '/*'))

# get number of images in training directory
nb_train_samples = 0
for r, dirs, files in os.walk(train_dir):
    for dr in dirs:
        nb_train_samples += len(glob.glob(os.path.join(r, dr + "/*")))

# get number of images in validation directory
nb_validate_samples = 0
for r, dirs, files in os.walk(validate_dir):
    for dr in dirs:
        nb_validate_samples += len(glob.glob(os.path.join(r, dr + "/*")))

print(nb_train_samples)
print(nb_validate_samples)
print(nb_classes)

# data pre-processing for training
train_datagen =  ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip = True)

# data pre-processing for validation
validate_datagen =  ImageDataGenerator(
    rescale = 1./255,
    rotation_range = 20,
    width_shift_range = 0.2,
    height_shift_range = 0.2,
    shear_range = 0.2,
    zoom_range = 0.2,
    fill_mode = 'nearest',
    horizontal_flip = True)

testdatagen=ImageDataGenerator(rescale=1./255)

train_gener =train_datagen.flow_from_directory(
train_dir,
target_size=(img_height,img_width),
#the total amount should be divisable by batch size
batch_size=5119,
shuffle=False,
class_mode= "categorical"
)

# generate and store training data
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size)

counter = Counter(train_generator.classes)                          
max_val = float(max(counter.values()))       

(train_pics,train_labels) = next(train_gener)
class_weights = calculating_class_weights(train_labels)

'''
class_weights = class_weight.compute_class_weight(
               'balanced',
                np.unique(train_generator.classes), 
                train_generator.classes)
'''

# generate and store validation data
validate_generator = validate_datagen.flow_from_directory(
    validate_dir,
    target_size = (img_width, img_height),
    batch_size = batch_size)

#%% define the model (see other evaluation_load file for the other models, and you can copy paste them here to use them)
xceptionmodel = Xception(
    include_top=False,
    weights="imagenet",
    input_shape=(img_width, img_height,3)
)
for layer in xceptionmodel.layers[:2]:
    layer.trainable=False
    
xceptionmodel.summary()

x= xceptionmodel.output
x = Flatten(name='flatten')(x)
x = Dense(512,activation='relu')(x)
x = Dropout(0.5)(x)
predictions= Dense(nb_classes,activation='softmax')(x)

model= Model(inputs=[xceptionmodel.input], outputs= predictions)


## Training

checkpoint = ModelCheckpoint( "Model/Balanced9_1/weights3.{epoch:02d}-{val_loss:.2f}.hdf5", verbose=1, save_best_only=True, save_weights_only=False, mode='auto', period=1)
model.compile(loss=get_weighted_loss(class_weights), optimizer=optimizers.Adam(learning_rate = 0.00005), metrics = ['accuracy'])

STEP_SIZE_TRAIN=train_generator.n//train_generator.batch_size
STEP_SIZE_VALID=validate_generator.n//validate_generator.batch_size
#STEP_SIZE_TEST=test_generator.n//test_generator.batch_size

metrics = Metrics(validate_generator, STEP_SIZE_VALID)

history = model.fit_generator(generator=train_generator,
                    steps_per_epoch=STEP_SIZE_TRAIN,
                    validation_data=validate_generator,
                    validation_steps=STEP_SIZE_VALID,
                    epochs=nb_epochs,
                callbacks=[metrics, checkpoint]
)

# save transfer learning model for offline prediction purposes
model.save('Model/Balanced9_1/flower_xception3_60_epoch.h5')

xfer_acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
xfer_loss = history.history['loss']
val_loss = history.history['val_loss']
epoch_list = range(len(xfer_acc))

f, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
t = f.suptitle('xception3', fontsize=12)
f.subplots_adjust(top=0.85, wspace=0.3)

ax1.plot(epoch_list, xfer_acc , label='Train Accuracy')
ax1.plot(epoch_list, val_acc, label='Validation Accuracy')
ax1.set_xticks(np.arange(0, 101, 10))
ax1.set_ylabel('Accuracy Value')
ax1.set_xlabel('Epoch')
ax1.set_title('Accuracy')
l1 = ax1.legend(loc="best")

ax2.plot(epoch_list, xfer_loss, label='Train Loss')
ax2.plot(epoch_list, val_loss, label='Validation Loss')
ax2.set_xticks(np.arange(0, 101, 10))
ax2.set_ylabel('Loss Value')
ax2.set_xlabel('Epoch')
ax2.set_title('Loss')
l2 = ax2.legend(loc="best")
plt.savefig('Model/Balanced9_1/CF1_Accuracy_Loss_VGG16_FC1_100epoch.png')


#Predict Training Data
dirName = 'Training';
listOfFiles = getListOfFiles(dirName)
listOfFiles = list()
for (dirpath, dirnames, filenames) in os.walk(dirName):
    listOfFiles += [os.path.join(dirpath, file) for file in filenames]
count = 0
for subdir, dirs, files in os.walk(dirName):
    for file_name in files:
        count += 1  
num_images = count
print(num_images)
predict_files = listOfFiles

im = cv2.imread(predict_files[0])
im = cv2.cvtColor(im, cv2.COLOR_BGR2RGB)
im = cv2.resize(im, (img_width, img_height)).astype(np.float32)
im = np.expand_dims(im, axis = 0)/255

predictor, image_id, pred4 = [], [], []
for i in predict_files:
    im = cv2.imread(i)
    im = cv2.resize(cv2.cvtColor(im, cv2.COLOR_BGR2RGB), (img_width, img_height)).astype(np.float32) / 255.0
    im = np.expand_dims(im, axis =0)
    pred1 = [model.predict(im)]
    pred2 = max(pred1)
    pred3 = max(max(pred2)).tolist()
    #print("Softmax:",pred3)
    pred4.append(pred3)
    outcome = [np.argmax(model.predict(im))]
    predictor.extend(list(outcome))
    image_id.extend([i.rsplit("/")[-1]])
    
final = pd.DataFrame()
final["id"] = image_id
final["Softmax"] = pred4
final["CL"] = predictor

classes = train_generator.class_indices
classes = {value : key for key, value in classes.items()}

final["CL"] = final["CL"].apply(lambda x: classes[x])
final.head(num_images)

final.to_csv("Model/Balanced9_1/train.csv", index=False)

#-------------------Predict Testing Data---------------------
dirName1 = 'Testing';
listOfFiles1 = getListOfFiles(dirName1)
listOfFiles1 = list()
for (dirpath, dirnames, filenames) in os.walk(dirName1):
    listOfFiles1 += [os.path.join(dirpath, file) for file in filenames]
count1 = 0
for subdir, dirs, files in os.walk(dirName1):
    for file_name in files:
        count1 += 1  
num_images1 = count1
print(num_images1)
predict_files1 = listOfFiles1

im1 = cv2.imread(predict_files1[0])
im1 = cv2.cvtColor(im1, cv2.COLOR_BGR2RGB)
im1 = cv2.resize(im1, (img_width, img_height)).astype(np.float32)
im1 = np.expand_dims(im1, axis = 0)/255

predictor1, image_id1, pred4_1 = [], [], []
for i in predict_files1:
    im1 = cv2.imread(i)
    im1 = cv2.resize(cv2.cvtColor(im1, cv2.COLOR_BGR2RGB), (img_width, img_height)).astype(np.float32) / 255.0
    im1 = np.expand_dims(im1, axis =0)
    pred1_1 = [model.predict(im1)]
    pred2_1 = max(pred1_1)
    pred3_1 = max(max(pred2_1)).tolist()
    #print("Softmax:",pred3)
    pred4_1.append(pred3_1)
    outcome = [np.argmax(model.predict(im1))]
    predictor1.extend(list(outcome))
    image_id1.extend([i.rsplit("/")[-1]])
    
final = pd.DataFrame()
final["id"] = image_id1
final["Softmax"] = pred4_1
final["Texture"] = predictor1

classes = train_generator.class_indices
classes = {value : key for key, value in classes.items()}

final["Texture"] = final["Texture"].apply(lambda x: classes[x])
final.head(num_images1)

final.to_csv("Model/Balanced9_1/test.csv", index=False)

#compute test accuracy
test_datagen = ImageDataGenerator(rescale = 1./255)
test_set = test_datagen.flow_from_directory('Testing',
                                            target_size = (img_width, img_height),
                                            batch_size = batch_size)
score = model.evaluate_generator(test_set, num_images1/batch_size)
print("Test Score:", score[0])
print("Test Accuracy:", score[1])



#------------------------Predict Validation Data-----------------------------------
dirName2 = 'Validation';
listOfFiles2 = getListOfFiles(dirName2)
listOfFiles2 = list()
for (dirpath, dirnames, filenames) in os.walk(dirName2):
    listOfFiles2 += [os.path.join(dirpath, file) for file in filenames]
count2 = 0
for subdir, dirs, files in os.walk(dirName2):
    for file_name in files:
        count2 += 1  
num_images2 = count2
print(num_images2)
predict_files2 = listOfFiles2

im2 = cv2.imread(predict_files2[0])
im2 = cv2.cvtColor(im2, cv2.COLOR_BGR2RGB)
im2 = cv2.resize(im2, (img_width, img_height)).astype(np.float32)
im2 = np.expand_dims(im2, axis = 0)/255

predictor2, image_id2, pred4_2 = [], [], []
for i in predict_files2:
    im2 = cv2.imread(i)
    im2 = cv2.resize(cv2.cvtColor(im2, cv2.COLOR_BGR2RGB), (img_width, img_height)).astype(np.float32) / 255.0
    im2 = np.expand_dims(im2, axis =0)
    pred1_2 = [model.predict(im2)]
    pred2_2 = max(pred1_2)
    pred3_2 = max(max(pred2_2)).tolist()
    #print("Softmax:",pred3)
    pred4_2.append(pred3_2)
    outcome = [np.argmax(model.predict(im2))]
    predictor2.extend(list(outcome))
    image_id2.extend([i.rsplit("/")[-1]])
    
final = pd.DataFrame()
final["id"] = image_id2
final["Softmax"] = pred4_2
final["Texture"] = predictor2

classes = train_generator.class_indices
classes = {value : key for key, value in classes.items()}

final["Texture"] = final["Texture"].apply(lambda x: classes[x])
final.head(num_images2)

final.to_csv("Model/Balanced9_1/val.csv", index=False)

#compute test accuracy
val_datagen = ImageDataGenerator(rescale = 1./255)
val_set = val_datagen.flow_from_directory('Validation',
                                            target_size = (img_width, img_height),
                                            batch_size = batch_size)
score1 = model.evaluate_generator(val_set, num_images2/batch_size)
print("Val Score:", score1[0])
print("Val Accuracy:", score1[1])

