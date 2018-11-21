import cv2
import numpy as np
import os
from cnn import LeNet
from sklearn.model_selection import train_test_split
import random
from keras.optimizers import SGD, Adam
from scipy import stats

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

def construct_train_test_data(input_h, input_w, channels, no_of_imgs, data, labels, test_size): #Randomly splits data into train and test sets (test_size must be between 0-1, ratio of test samples)
    rand_interval = 1/test_size
    test_sample_index = 0
    test_index = 0
    train_index = 0
    test_size = int(test_size*no_of_imgs)
    train_size = no_of_imgs - test_size
    train_data = np.zeros(shape=(train_size,input_h,input_w, channels))
    test_data = np.zeros(shape=(test_size,input_h,input_w, channels))
    test_labels = np.zeros(shape=(test_size,1))
    train_labels = np.zeros(shape=(train_size,1))
    print(rand_interval)
    for x in range(len(labels)):
        if x % rand_interval == 0:
            test_sample_index = random.randint(x,x+rand_interval-1)
            print("random interval is from {} to {}".format(x, x+rand_interval-1))
            print("test sample index {}".format(test_sample_index))
            print("test index {}".format(test_index))
            test_data[test_index,:,:,:] =  data[test_sample_index,:,:,:]
            test_labels[test_index] = labels[test_sample_index]
            test_index += 1
        if x != test_sample_index:
            train_data[train_index,:,:,:] = data[x,:,:,:]
            print("x {}".format(x))
            print("train index {}".format(train_index))
            train_labels[train_index] = labels[x]
            train_index += 1
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/train_data', train_data)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/train_label', train_labels)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/test_data', test_data)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/test_label', test_labels)
    return train_data, test_data, train_labels, test_labels


def construct_train_test_data_grayscale(input_h, input_w, no_of_imgs, data, labels, test_size): #Randomly splits data into train and test sets (test_size must be between 0-1, ratio of test samples)
    rand_interval = 1/test_size
    test_sample_index = 0
    test_index = 0
    train_index = 0
    test_size = int(test_size*no_of_imgs)
    train_size = no_of_imgs - test_size
    train_data = np.zeros(shape=(train_size,input_h,input_w))
    test_data = np.zeros(shape=(test_size,input_h,input_w))
    test_labels = np.zeros(shape=(test_size,1))
    train_labels = np.zeros(shape=(train_size,1))
    print(rand_interval)
    for x in range(len(labels)):
        if x % rand_interval == 0:
            test_sample_index = random.randint(x,x+rand_interval-1)
            print("random interval is from {} to {}".format(x, x+rand_interval-1))
            print("test sample index {}".format(test_sample_index))
            print("test index {}".format(test_index))
            test_data[test_index,:,:] =  data[test_sample_index,:,:]
            test_labels[test_index] = labels[test_sample_index]
            test_index += 1
        if x != test_sample_index:
            train_data[train_index,:,:] = data[x,:,:]
            print("x {}".format(x))
            print("train index {}".format(train_index))
            train_labels[train_index] = labels[x]
            train_index += 1
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/train_data', train_data)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/train_label', train_labels)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/test_data', test_data)
    np.save('/Users/iremergun/Desktop/ucr_classes/cs235/proj/test_label', test_labels)
    return train_data, test_data, train_labels, test_labels
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
opt = SGD(lr=0.1)
model.compile(loss="sparse_categorical_crossentropy", optimizer=opt, metrics=["accuracy"]) #categorical_crossentropy
#train_data = train_data.reshape((train_data.shape[0], 324, 300, 1))
#test_data = test_data.reshape((test_data.shape[0], 324, 300, 1))
print("[INFO] training...")
model.fit(train_data, train_label, batch_size=64, epochs=30, verbose=1)
(loss, accuracy) = model.evaluate(test_data, test_label, batch_size=64, verbose=1)
print("[INFO] accuracy: {:.2f}%".format(accuracy * 100))
