import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense, Conv2D, MaxPooling2D, Flatten
from keras.models import Model
from keras.applications.vgg16 import VGG16
from keras.applications.imagenet_utils import decode_predictions
from keras.preprocessing.image import ImageDataGenerator
from keras.applications.vgg19 import VGG19
from keras.applications.imagenet_utils import decode_predictions
import seaborn as sn
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import classification_report, confusion_matrix
from peice_predictor import predict


folder = '/content/drive/MyDrive/splitted_data'
image_size = (224, 224) #desired input size of image
batch_size = 32
test_datagen = ImageDataGenerator(rescale=1./255)

datagen = ImageDataGenerator(
    rotation_range=5,
    # width_shift_range=0.1,
    # height_shift_range=0.1,
    rescale=1./255,
    horizontal_flip=True,
    fill_mode='nearest')


#Train Data preprocessing
train_gen = datagen.flow_from_directory(
    folder + '/train',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=True
)

#Test data preprocessing
test_gen = test_datagen.flow_from_directory(
    folder + '/test',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

#model1 with VGG16 architecture

base_model = VGG16(weights='imagenet', include_top=False,
                   input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in base_model.layers:
    layer.trainable = False

# Establish new fully connected block
x = base_model.output
x = Flatten()(x)  # flatten from convolution tensor output
# number of layers and units are hyperparameters, as usual
x = Dense(500, activation='relu')(x)
x = Dense(500, activation='relu')(x)
predictions = Dense(13, activation='softmax')(
    x)  # should match # of classes predicted

# this is the model we will train
model = Model(inputs=base_model.input, outputs=predictions)
model.compile(optimizer='adam', loss='categorical_crossentropy',
              metrics=['categorical_accuracy'])

model.summary()


epochs = 10

history = model.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    validation_data=test_gen
)
model.save_weights('model_VGG16.h5')


#accuracy vs training epochs
plt.plot(history.history['categorical_accuracy'], 'ko')
plt.plot(history.history['val_categorical_accuracy'], 'b')

plt.title('Accuracy vs Training Epoch')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(['Train', 'Validation'])


#testing model for test data
test_gen.reset()
Y_pred = model.predict_generator(test_gen)
classes = test_gen.classes[test_gen.index_array]
y_pred = np.argmax(Y_pred, axis=-1)
# print(sum(y_pred==classes)/810)

target_names = ['BB', 'BK', 'BN', 'BP', 'BQ', 'BR',
                'Empty', 'WB', 'WK', 'WN', 'WP', 'WQ', 'WR']

print(classification_report(
    test_gen.classes[test_gen.index_array], y_pred, target_names=target_names))


#model 2 with VGG19 architecture 
base_model_two = VGG19(
    weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in base_model_two.layers:
    layer.trainable = False

# Establish new fully connected block
x = base_model_two.output
x = Flatten()(x)  # flatten from convolution tensor output
# number of layers and units are hyperparameters, as usual
x = Dense(500, activation='relu')(x)
x = Dense(500, activation='relu')(x)
predictions = Dense(13, activation='softmax')(
    x)  # should match # of classes predicted

# this is the model we will train
model_two = Model(inputs=base_model_two.input, outputs=predictions)
#compiling the model
model_two.compile(optimizer='adam',
                  loss='categorical_crossentropy', metrics=['accuracy'])

model_two.summary()

epochs = 10
#training the model

history = model_two.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    validation_data=test_gen
)
model.save_weights('model_VGG19.h5')

#testing model with test data
test_gen.reset()
Y_pred = model_two.predict_generator(test_gen)
classes = test_gen.classes[test_gen.index_array]
y_pred = np.argmax(Y_pred, axis=-1)

target_names = ['BB', 'BK', 'BN', 'BP', 'BQ', 'BR',
                'Empty', 'WB', 'WK', 'WN', 'WP', 'WQ', 'WR']

print(classification_report(
    test_gen.classes[test_gen.index_array], y_pred, target_names=target_names))


#model3 vgg16 architecture with variation in fully connected layer
model3 = VGG16(weights='imagenet', include_top=False,
               input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in model3.layers:
    layer.trainable = False

# Establish new fully connected block
x = model3.output
x = Flatten()(x)  # flatten from convolution tensor output
# number of layers and units are hyperparameters, as usual
x = Dense(500, activation='sigmoid')(x)
x = Dense(500, activation='sigmoid')(x)
predictions = Dense(13, activation='softmax')(
    x)  # should match # of classes predicted


# this is the model we will train
model3 = Model(inputs=model3.input, outputs=predictions)
model3.compile(optimizer='adam', loss='categorical_crossentropy',
               metrics=['categorical_accuracy'])

model3.summary()

epochs = 10
#training the model
history = model3.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    validation_data=test_gen
)
model3.save_weights('model3_s.h5')

#testing the model with test data.
test_gen.reset()
Y_pred = model3.predict_generator(test_gen)
classes = test_gen.classes[test_gen.index_array]
y_pred = np.argmax(Y_pred, axis=-1)
target_names = ['BB', 'BK', 'BN', 'BP', 'BQ', 'BR',
                'Empty', 'WB', 'WK', 'WN', 'WP', 'WQ', 'WR']


print(classification_report(
    test_gen.classes[test_gen.index_array], y_pred, target_names=target_names))



#model4 with VGG19 architecture with varied fully connected layer
model4 = VGG19(weights='imagenet', include_top=False,
               input_shape=(224, 224, 3))

# Freeze convolutional layers
for layer in model4.layers:
    layer.trainable = False

# Establish new fully connected block
x = model4.output
x = Flatten()(x)  # flatten from convolution tensor output
# number of layers and units are hyperparameters, as usual
x = Dense(500, activation='sigmoid')(x)
x = Dense(500, activation='sigmoid')(x)
predictions = Dense(13, activation='softmax')(
    x)  # should match # of classes predicted


# this is the model we will train
model4 = Model(inputs=model4.input, outputs=predictions)
#compiling the model
model4.compile(optimizer='adam', loss='categorical_crossentropy',
               metrics=['categorical_accuracy'])

model4.summary()

epochs = 10
#training the model
history = model4.fit(
    train_gen,
    epochs=epochs,
    verbose=1,
    validation_data=test_gen
)
model4.save_weights('model4_s.h5')

#testing with test data
test_gen.reset()
Y_pred = model4.predict_generator(test_gen)
classes = test_gen.classes[test_gen.index_array]
y_pred = np.argmax(Y_pred, axis=-1)
target_names = ['BB', 'BK', 'BN', 'BP', 'BQ', 'BR',
                'Empty', 'WB', 'WK', 'WN', 'WP', 'WQ', 'WR']


print(classification_report(
    test_gen.classes[test_gen.index_array], y_pred, target_names=target_names))





#testing for given chess board and predicting peices in every square.
new_data = test_datagen.flow_from_directory(
    '/content/drive/MyDrive/cropped_images/',
    target_size=image_size,
    batch_size=batch_size,
    class_mode='categorical',
    color_mode='rgb',
    shuffle=False
)

new_data.reset()
Y_pred = model.predict_generator(new_data)
classes = new_data.classes[new_data.index_array]
y_pred = np.argmax(Y_pred, axis=-1)


for i in range(64):
    print(i,predict[i])