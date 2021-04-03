from keras.models import Model
from keras.layers import Flatten
from keras.layers import Dense
from keras.layers import Input
from keras.layers import Conv2D
from keras.layers import MaxPooling2D, MaxPool2D

img_input = Input((224,224,3))


x = Conv2D(64, (3, 3), activation='relu', padding='same',
           name='block1_conv1')(img_input)
x = Conv2D(64, (3, 3), activation='relu',
           padding='same', name='block1_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

# Block 2
x = Conv2D(128, (3, 3), activation='relu',
           padding='same', name='block2_conv1')(x)
x = Conv2D(128, (3, 3), activation='relu',
           padding='same', name='block2_conv2')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

# Block 3
x = Conv2D(256, (3, 3), activation='relu',
           padding='same', name='block3_conv1')(x)
x = Conv2D(256, (3, 3), activation='relu',
           padding='same', name='block3_conv2')(x)
x = Conv2D(256, (3, 3), activation='relu',
           padding='same', name='block3_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

# Block 4
x = Conv2D(512, (3, 3), activation='relu',
           padding='same', name='block4_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu',
           padding='same', name='block4_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu',
           padding='same', name='block4_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

# Block 5
x = Conv2D(512, (3, 3), activation='relu',
           padding='same', name='block5_conv1')(x)
x = Conv2D(512, (3, 3), activation='relu',
            padding='same', name='block5_conv2')(x)
x = Conv2D(512, (3, 3), activation='relu',
            padding='same', name='block5_conv3')(x)
x = MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

# Classification block
x = Flatten(name='flatten')(x)
x = Dense(500, activation='relu', name='fc1')(x)
x = Dense(500, activation='relu', name='fc2')(x)
x = Dense(13, activation='softmax', name='predictions')(x)


model1 = Model(input=img_input, outputs=x)

#now our model is ready we should compile and train it