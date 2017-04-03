import csv
import cv2
import numpy as np

from keras.models import Sequential
from keras.layers import Dense, Flatten, Lambda, Convolution2D, Cropping2D, Dropout

from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

correction_factor = 0.3
batch_size = 256


def process_image(full_path, crop_size=None, image_size=(160, 320)):
    """
    This method takes an image path, localize it for E2C processing.

    :param full_path: The captured path from simulator
    :param crop_size: (top, bottom)
    :param image_size: (height, width)
    :return: np.array containing image data        
    """
    image_path = "./IMG/" + full_path.split("/")[-1]
    this_image = cv2.imread(image_path)
    if crop_size is not None:
        this_image = this_image[crop_size[0]:image_size[0]-crop_size[1]]
    return np.array(this_image)


def generator(samples, batch_size=32):
    """
    Generator to batch process training and validation data
    
    :param samples: 
    :param batch_size: 
    :return: batch data 
    """
    print("In generator #############")
    while True:
        shuffle(samples)
        for index in range(0, len(samples), batch_size):
            batch_sample = samples[index:index + batch_size]
            images = []
            measures = []
            for item in batch_sample:
                center_image = process_image(item[0])
                left_image = process_image(item[1])
                right_image = process_image(item[2])

                center_measure = float(item[3])
                right_measure = center_measure - correction_factor
                left_measure = center_measure + correction_factor

                images.extend([left_image, center_image, right_image])
                measures.extend([left_measure, center_measure, right_measure])

            x_train = np.array(images)
            y_train = np.array(measures)
            yield shuffle(x_train, y_train)


def create_model():
    """
    Create main model primarily 
    :return: 
    """
    model = Sequential()
    model.add(Lambda(lambda x: x / 255 - 0.5, input_shape=(160, 320, 3)))

    model.add(Cropping2D(cropping=((70, 25), (0, 0))))

    model.add(Convolution2D(24, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(36, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(48, 5, 5, subsample=(2, 2), activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Convolution2D(64, 3, 3, activation="relu"))
    model.add(Flatten())
    model.add(Dense(100))
    model.add(Dense(50))
    model.add(Dense(10))
    model.add(Dense(1))

    model.compile(loss="mse", optimizer="adam")
    return model


def create_lenet():
    pass


# ------------ Main Execution Path --------------------

# >> Read all the samples from the driving log
driving_log = []
with open("driving_log.csv") as f:
    reader = csv.reader(f)
    for log in reader:
        driving_log.append(log)

# >> Splitting data into 75-25 for training and testing
train_sample, validation_sample = train_test_split(driving_log, test_size=0.25)

# >> Create generators for training and validation samples
train_generator = generator(train_sample, batch_size=batch_size)
validation_generator = generator(validation_sample, batch_size=batch_size)

# >> Create Model and execute
model = create_model()

# model.fit(X_train, y_train, validation_split=0.2, shuffle=True, epochs=3, verbose=1)
model.fit_generator(train_generator,
                    steps_per_epoch=len(train_sample)/batch_size,
                    validation_data=validation_generator,
                    validation_steps=len(validation_sample)/batch_size,
                    epochs=3)

# >> Saving for drive
model.save("model.h5")


