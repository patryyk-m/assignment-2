from __future__ import print_function

import os

import keras
import tensorflow as tf
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten, Conv2D, MaxPooling2D, Rescaling, BatchNormalization
from keras.optimizers import RMSprop,Adam
import matplotlib.pyplot as plt
import numpy as np


batch_size = 12
num_classes = 3
epochs = 15
img_width = 128
img_height = 128
img_channels = 3
fit = True #make fit false if you do not want to train the network again
_data_root = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'chest_xray')
train_dir = os.path.join(_data_root, 'train')
test_dir = os.path.join(_data_root, 'test')

with tf.device('/gpu:0'):
    
    #create training,validation and test datatsets
    train_ds,val_ds = tf.keras.preprocessing.image_dataset_from_directory(
        train_dir,
        seed=123,
        validation_split=0.2,
        subset='both',
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)
    
    test_ds = tf.keras.preprocessing.image_dataset_from_directory(
        test_dir,
        seed=None,
        image_size=(img_height, img_width),
        batch_size=batch_size,
        labels='inferred',
        shuffle=True)

    class_names = train_ds.class_names
    print('Class Names: ',class_names)
    num_classes = len(class_names)

    AUTOTUNE = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(buffer_size=AUTOTUNE)
    val_ds = val_ds.prefetch(buffer_size=AUTOTUNE)
    test_ds = test_ds.prefetch(buffer_size=AUTOTUNE)

    data_augmentation = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal"),
        tf.keras.layers.RandomRotation(0.05),
        tf.keras.layers.RandomZoom(0.1),
        tf.keras.layers.RandomContrast(0.1),
    ])
    
    plt.figure(figsize=(10, 10))
    for images, labels in train_ds.take(2):
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            plt.title(class_names[labels[i].numpy()])
            plt.axis("off")
    plt.savefig("run3_samples.png", bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2)
    plt.close()

    #create model
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=(img_height, img_width, img_channels)),
        data_augmentation,
        Rescaling(1.0/255),
        Conv2D(32, (3,3), activation = 'relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation = 'relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        Conv2D(64, (3,3), activation = 'relu'),
        BatchNormalization(),
        MaxPooling2D(2,2),
        tf.keras.layers.GlobalAveragePooling2D(), # reduces each feature map to a single value
        Dense(256, activation = 'relu'),
        BatchNormalization(),
        Dropout(0.5),
        Dense(num_classes, activation = 'softmax')
    ])

    model.compile(loss='sparse_categorical_crossentropy',
                  optimizer=Adam(learning_rate=0.0005),
                  metrics=['accuracy'])
    
    earlystop_callback = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=4, restore_best_weights=True)
    save_callback = tf.keras.callbacks.ModelCheckpoint("pneumonia.keras",save_freq='epoch',save_best_only=True)

    if fit:
        history = model.fit(
            train_ds,
            batch_size=batch_size,
            validation_data=val_ds,
            callbacks=[save_callback, earlystop_callback],
            epochs=epochs)
    else:
        model = tf.keras.models.load_model("pneumonia.keras")

    #if shuffle=True when creating the dataset, samples will be chosen randomly   
    score = model.evaluate(test_ds, batch_size=batch_size)
    print('Test accuracy:', score[1])

    
    if fit:
        plt.figure()
        plt.plot(history.history['accuracy'])
        plt.plot(history.history['val_accuracy'])
        plt.title('model accuracy')
        plt.ylabel('accuracy')
        plt.xlabel('epoch')
        plt.legend(['train', 'val'], loc='upper left')
        plt.savefig("run3_accuracy.png", bbox_inches="tight")
        plt.show(block=False)
        plt.pause(2)
        plt.close()
        
    test_batch = test_ds.take(1)
    plt.figure(figsize=(10, 10))
    for images, labels in test_batch:
        for i in range(6):
            ax = plt.subplot(2, 3, i + 1)
            plt.imshow(images[i].numpy().astype("uint8"))
            prediction = model.predict(tf.expand_dims(images[i].numpy(),0), verbose=0)#perform a prediction on this image
            plt.title('Actual:' + class_names[labels[i].numpy()]+ '\nPredicted:{} {:.2f}%'.format(class_names[np.argmax(prediction)], 100 * np.max(prediction)))
            plt.axis("off")
    plt.savefig("run3_predictions.png", bbox_inches="tight")
    plt.show(block=False)
    plt.pause(2)
    plt.close()