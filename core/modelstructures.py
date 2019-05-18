from keras import Model
from keras.applications import InceptionResNetV2, InceptionV3, VGG16, MobileNet
from keras.layers import GlobalAveragePooling2D, Dropout, Dense


HEIGHT = WIDTH = 224

def model_inception_resnetV2():
    model = InceptionResNetV2(include_top = False, input_shape = (HEIGHT, WIDTH, 3), weights='imagenet')
    addition = GlobalAveragePooling2D()(model.output)
    addition = Dropout(0.7)(addition)
    addition = Dense(256,activation='relu')(addition)
    addition = Dense(3, activation='softmax')(addition)
    model = Model(model.inputs, addition)
    return model


def model_inception_V3():
    model = InceptionV3(include_top=False, input_shape=(HEIGHT, WIDTH, 3), weights='imagenet')
    addition = GlobalAveragePooling2D()(model.output)
    addition = Dropout(0.6)(addition)
    addition = Dense(128, activation='relu')(addition)
    addition = Dense(3, activation='softmax')(addition)
    model = Model(model.inputs, addition)
    return model


def model_VGG16():
    model = VGG16(include_top=False, weights = None, input_shape=(HEIGHT, WIDTH, 3))
    addition = GlobalAveragePooling2D()(model.output)
    addition = Dropout(0.5)(addition)
    addition = Dense(256, activation='relu')(addition)
    addition = Dense(3, activation='softmax')(addition)
    model = Model(model.inputs, addition)
    return model


def model_mobilenet():
    # Create a MobileNet model
    mobile = MobileNet()

    # Modify the model
    # Choose the 6th layer from the last
    x = mobile.layers[-6].output

    # Add a dropout and dense layer for predictions
    x = Dropout(0.25)(x)
    predictions = Dense(7, activation='softmax')(x)

    # Create a new model with the new outputs
    model = Model(inputs=mobile.input, outputs=predictions)

    # Prevent everything except the last 23 layers from being trained
    for layer in model.layers[:-23]:
        layer.trainable = False

    return model