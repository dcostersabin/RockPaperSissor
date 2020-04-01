import os
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.preprocessing.image import ImageDataGenerator
from keras.models import load_model
from keras.preprocessing.image import image
import numpy as np

BASE = os.getcwd()
TOTAL_DATASET = BASE + '/Datasets/TotalDatasets/'
TRAINING_SET = BASE + '/Datasets/TraningSet/'
TEST_SET = BASE + '/Datasets/TestSet/'
TEMP = [TRAINING_SET, TEST_SET]
# define train test split
TOTAL_TRAINING_DATA = 0.8
# define dir for saving model
MODEL_DIR = BASE + '/Model/'
NO_OF_EPOCH = 25
TESTING_MODEL = TRAINING_SET + 'rock/2affjOmZChc9AXpR.png'


def main():
    make_framework()
    train_test_split()
    if check_for_model():
        classifier = load_model(MODEL_DIR + '/myModel')
        load(classifier)
    else:
        print("Do You Want To Run Neural Network ")
        decision = input("Press 'Y' If Yes")
        if decision == 'Y':
            run()
        else:
            print("Exiting The Program")


def get_avg_files(dir_name):
    file_counter = 0
    total_count = 0
    for dirs in os.listdir(dir_name):
        file_counter += len(os.listdir(str(dir_name + dirs)))
        total_count += 1
    return int(file_counter / total_count)


# get the main dir where the images of 3 types are stored withhin TOTAL_DATASET for further classification
def get_dirs():
    types = os.listdir(TOTAL_DATASET)
    return types


# Creating directories
def make_dirs(folder_name):
    try:
        os.makedirs(folder_name)
    except FileExistsError:
        print("The Folder Already Exists", folder_name)


# make overall folder structure for further classifications
def make_framework():
    make_dirs(TRAINING_SET)
    make_dirs(TEST_SET)
    make_dirs(MODEL_DIR)
    for dirname in TEMP:
        print("Creating Folder In", dirname)
        for subdir in get_dirs():
            make_dirs(str(dirname + '/' + str(subdir)))


# split  images placed in totaldatset to train and test split
def train_test_split():
    print("Processing Images")
    training = test = 0
    for dirs in get_dirs():
        total_data = len(os.listdir(str(TOTAL_DATASET + '/' + str(dirs))))
        print("Total Files in ", dirs, 'is', total_data)
        counter = 0
        CURRENT_DIR = ''
        for images in os.listdir(str(TOTAL_DATASET + '/' + str(dirs))):
            if counter <= (total_data * TOTAL_TRAINING_DATA):
                CURRENT_DIR = TRAINING_SET
                training += 1
            else:
                CURRENT_DIR = TEST_SET
                test += 1
            os.rename(TOTAL_DATASET + str(dirs) + '/' + str(images), CURRENT_DIR + str(dirs) + '/' + str(images))
            counter += 1
            print("Moving file", images, 'to', CURRENT_DIR + '/' + str(dirs))


def run():
    # initializing the model
    classifier = Sequential()
    # adding convolution  layer
    classifier.add(Convolution2D(64, 3, 3, input_shape=(150, 150, 3), activation='relu'))
    # adding max pooling layer
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # adding another convolution layer
    classifier.add(Convolution2D(64, 3, 3, input_shape=(150, 150, 3), activation='relu'))
    # adding next pooling layer for convolution layer
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # adding next convolution layer
    classifier.add(Convolution2D(64, 3, 3, input_shape=(150, 150, 3), activation='relu'))
    # adding one more pooling
    classifier.add(MaxPooling2D(pool_size=(2, 2)))
    # adding a flatten layer
    classifier.add(Flatten())
    # adding the last dense layer
    classifier.add(Dense(512, activation='relu'))
    # adding final layer for output
    classifier.add(Dense(3, activation='softmax'))
    # compiling the neural network
    classifier.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])
    # converting the images to preferred input size using keras
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True)

    test_datagen = ImageDataGenerator(rescale=1. / 255)

    training_set = train_datagen.flow_from_directory(
        TRAINING_SET,
        target_size=(150, 150),
        batch_size=25,
        class_mode='categorical')

    test_set = test_datagen.flow_from_directory(
        TEST_SET,
        target_size=(150, 150),
        batch_size=25,
        class_mode='categorical')
    STEPS_IN_EPOCH_TRAINING = get_avg_files(TRAINING_SET)
    STEPS_IN_EPOCH_TEST = get_avg_files(TEST_SET)
    classifier.fit_generator(
        training_set,
        steps_per_epoch=STEPS_IN_EPOCH_TRAINING,
        epochs=NO_OF_EPOCH,
        validation_data=test_set,
        validation_steps=STEPS_IN_EPOCH_TEST)
    classifier.save(MODEL_DIR + 'myModel', overwrite=True)
    print("Model Trained And Saved At", MODEL_DIR)


def load(classifier):
    test_image = image.load_img(TESTING_MODEL, target_size=(150, 150))
    test_image = image.img_to_array(test_image)
    test_image = np.expand_dims(test_image, axis=0)
    prediction = classifier.predict(test_image)
    # print("The Prediction is ", prediction)
    if prediction[0][0] == 1:
        print("Paper")
    elif prediction[0][1] == 1:
        print("Rock")
    elif prediction[0][2] == 1:
        print("Scissors")


def check_for_model():
    return os.path.exists(MODEL_DIR + 'myModel')


if __name__ == '__main__':
    main()
