import cv2
import numpy as np
import os
from cnn import LeNet
from sklearn.model_selection import train_test_split
import random
from keras.optimizers import SGD, Adam
from scipy import stats
from keras.callbacks import EarlyStopping
from keras.utils import plot_model
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import itertools

genres =	{
  "metal": 0,
  "pop": 1,
  "disco": 2,
  "blues":3,
  "classical":4,
  "reggae":5,
  "rock":6,
  "hiphop":7,
  "country":8,
  "jazz":9
}

def construct_input_matrix(input_loc, input_h, input_w, channels, no_of_imgs):
    a = np.zeros(shape=(no_of_imgs,input_h,input_w, channels))
    labels = np.zeros(shape=(no_of_imgs,1))
    index = 0
    root_to_save = input_loc
    for root, dirs, files in os.walk(root_to_save):
        for directory in dirs:
            for root1, dirs1, files1 in os.walk(os.path.join(root, directory)):
                for f in files1:
                    if directory in genres:
                        labels[index] = genres[directory]
                    img = cv2.imread(os.path.join(root1,f),cv2.IMREAD_COLOR) 
                    a[index,:,:,:] = img
                    index += 1
                    '''
                    if index % 10 == 0 and index != 0:
                        break
                    '''
    a = stats.zscore(a)
    a = np.nan_to_num(a)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/data', a)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/labels', labels)
    return a, labels

def plot_confusion_matrix(cm, classes,
                          normalize=False,
                          title='Confusion matrix',
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(classes))
    plt.xticks(tick_marks, classes, rotation=45)
    plt.yticks(tick_marks, classes)

    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        plt.text(j, i, format(cm[i, j], fmt),
                 horizontalalignment="center",
                 color="white" if cm[i, j] > thresh else "black")

    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.tight_layout()


def construct_input_matrix_grayscale(input_loc, input_h, input_w, no_of_imgs):
    a = np.zeros(shape=(no_of_imgs,input_h,input_w))
    labels = np.zeros(shape=(no_of_imgs,1))
    index = 0
    root_to_save = input_loc
    for root, dirs, files in os.walk(root_to_save):
        for directory in dirs:
            for root1, dirs1, files1 in os.walk(os.path.join(root, directory)):
                for f in files1:
                    if directory in genres:
                        labels[index] = genres[directory]
                    img = cv2.imread(os.path.join(root1,f),cv2.IMREAD_GRAYSCALE) 
                    a[index,:,:] = img
                    index += 1
                    '''
                    if index % 10 == 0 and index != 0:
                        break
                    '''
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/data', a)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/labels', labels)
    return a, labels

'''
data,labels = construct_input_matrix('/Users/iremergun/Desktop/ucr_classes/cs235/proj/genres_tripled',324,300,3,3000)
print(data.shape)
print(data)
print(labels)

data = np.load('/Users/iremergun/Desktop/ucr_classes/cs235/proj/data.npy')
labels = np.load('/Users/iremergun/Desktop/ucr_classes/cs235/proj/labels.npy')
print(data)

#train_data, test_data, train_label, test_label = construct_train_test_data_grayscale(324,300,len(labels),data, labels, 0.1)

train_data, test_data, train_label, test_label = train_test_split(data, labels, test_size=0.10)
np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/train_data', train_data)
np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/train_label', train_label)
np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/test_data', test_data)
np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/test_label', test_label)
print(train_data.shape)
print(test_data.shape)
print(test_label)
print(train_label)
'''
train_data = np.load('/Users/iremergun/Desktop/ucr_classes/cs235/proj/train_data.npy') 
test_data = np.load('/Users/iremergun/Desktop/ucr_classes/cs235/proj/test_data.npy') 
train_label = np.load('/Users/iremergun/Desktop/ucr_classes/cs235/proj/train_label.npy') 
test_label = np.load('/Users/iremergun/Desktop/ucr_classes/cs235/proj/test_label.npy') 
print(train_data.shape)
print(test_data.shape)


model = LeNet.build_model_lenet(324,300,3,10)
opt = Adam(lr=0.001)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) #categorical_crossentropy
#train_data = train_data.reshape((train_data.shape[0], 324, 300, 1))
#test_data = test_data.reshape((test_data.shape[0], 324, 300, 1))
earlystop = EarlyStopping(monitor='val_acc', min_delta=0.0001, patience=8, \
                          verbose=1, mode='auto')
callbacks_list = [earlystop]
#plot_model(model, to_file='/Users/iremergun/Desktop/ucr_classes/cs235/proj/report/model.png', show_shapes=True)

print("[INFO] training...")
model.fit(train_data, train_label, batch_size=64, epochs=10, verbose=1, validation_data=(test_data, test_label) )
(loss, accuracy) = model.evaluate(test_data, test_label, batch_size=64, verbose=1)
results = model.predict_classes(test_data, batch_size=64, verbose =1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
print("loss: {}".format(loss))
res = confusion_matrix(test_label, results, labels=[0,1,2,3,4,5,6,7,8,9])#"metal", "pop", "disco", "blues", "classical", "reggae", "rock", "hiphop", "country", "jazz"]) 
class_names = [0,1,2,3,4,5,6,7,8,9]
plot_confusion_matrix(res, class_names)
plt.show()