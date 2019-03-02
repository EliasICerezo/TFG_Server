import random
from keras import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau,EarlyStopping
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.applications.inception_resnet_v2 import InceptionResNetV2
from keras.applications.inception_v3 import InceptionV3
import os
from glob import glob
from keras.utils import to_categorical
from keras.models import load_model
from keras.models import Sequential
from keras.preprocessing.image import ImageDataGenerator

csvpath = 'D:\OneDrive\TFG\TFG_Python\HAM10000_metadata.csv'
impath = 'D:\OneDrive\TFG\TFG_Python\Images'
h5path = 'D:\OneDrive\TFG\TFG_Python\core\model.h5'
#Constante que da las dimensiones para las distintas imagenes
HEIGHT = WIDTH = 224
LABELNUM = 7

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(impath, '*.jpg'))}
lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}


def model_training():
    #Local funcions definitions
    def test_gen():
        imglist = np.zeros(len(test_img_path), HEIGHT,WIDTH,3)
        labellist = np.zeros (len(test_img_path, len(labelnum)))
        for i, imgpath in enumerate(test_img_path):
            img = read_img(imgpath)
            label = valid_label[i]
            imglist[i] = img
            labellist[i] = label
        return (imglist, labellist)

    def valid_gen():
        imglist = np.zeros((len(valid_img_path), HEIGHT, WIDTH, 3))
        labellist = np.zeros((len(valid_img_path), len(labelnum)))
        for i, imgpath in enumerate(valid_img_path):
            img = read_img(imgpath)
            label = valid_label[i]
            imglist[i] = img
            labellist[i] = label
        return (imglist, labellist)

    def train_gen(batch_size=16):
        while (True):
            imglist = np.zeros((batch_size*2, HEIGHT, WIDTH, 3))
            labellist = np.zeros((batch_size*2, len(labelnum)))
            for i in range(batch_size):
                rndid = random.randint(0, len(train_img_path) - 1)
                imgpath = train_img_path[train_img_path.index[rndid]]
                img = read_img(imgpath)
                label = train_label[rndid]
                imglist[i*2] = img
                labellist[i*2] = label
                noise = np.random.random((HEIGHT, WIDTH, 3))
                imglist[(i * 2) + 1] = img + noise
                labellist[(i * 2) + 1] = label
            yield (imglist, labellist)
    
    #End of local funcion definitions

    train_df = pd.read_csv(csvpath)
    train_df['path'] = train_df['image_id'].map(imageid_path_dict.get)

    labelnum = train_df.groupby('dx').size()

    encoder = LabelEncoder()  # Lo usaremos para las labels
    encoder.fit(train_df['dx'])
    labeldf = encoder.transform(train_df['dx'])
    labeldf = to_categorical(labeldf)

    train_target, test_img_path, train_label, test_label = train_test_split(train_df['path'], labeldf, test_size=0.2)

    train_img_path, valid_img_path, train_label, valid_label = train_test_split(train_target, train_label, test_size=0.2)

    validdata = valid_gen()


    print("Creando modelo y compilandolo")
    model = model_inception_resnetV2()
    model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='Adam')
    callbacks = [
        #ModelCheckpoint('model.h5',monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
    ]
    print('Se comienza el entrenamiento del modelo')
    model.fit_generator(train_gen(), epochs=50, steps_per_epoch=80, verbose=2, validation_data=validdata, callbacks= callbacks)
    print("Entrenamiento completado, se procede al test final")
    test_x, test_y = test_gen()
    model.evaluate(test_x, test_y, verbose=1)





def read_img(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, dsize=(WIDTH, HEIGHT))
    return img



def model_inception_resnetV2():
    model = InceptionResNetV2(include_top = False, input_shape = (HEIGHT, WIDTH, 3), weights='imagenet')
    addition = GlobalAveragePooling2D()(model.output)
    addition = Dropout(0.5)(addition)
    addition = Dense(256, activation='relu')(addition)
    addition = Dense(LABELNUM, activation='softmax')(addition)
    model = Model(model.inputs, addition)
    return model

def model_inception_V3():
    model = InceptionV3(include_top=False, input_shape = (HEIGHT, WIDTH, 3), weights='imagenet', pooling = 'avg')
    addition = Dense(256, activation='relu')(model.output)
    addition = Dense(LABELNUM, activation='softmax')(addition)
    model = Model(model.inputs, addition)
    return model

def model_AlexNet():
    # Instantiate an empty model
    model = Sequential()

    # 1st Convolutional Layer
    model.add(Conv2D(filters=96, input_shape=(224, 224, 3), kernel_size=(11, 11), strides=(4, 4), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 2nd Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(11, 11), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # 3rd Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 4th Convolutional Layer
    model.add(Conv2D(filters=384, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))

    # 5th Convolutional Layer
    model.add(Conv2D(filters=256, kernel_size=(3, 3), strides=(1, 1), padding='valid'))
    model.add(Activation('relu'))
    # Max Pooling
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2), padding='valid'))

    # Passing it to a Fully Connected layer
    model.add(Flatten())
    # 1st Fully Connected Layer
    model.add(Dense(4096, input_shape=(224 * 224 * 3,)))
    model.add(Activation('relu'))
    # Add Dropout to prevent overfitting
    model.add(Dropout(0.4))

    # 2nd Fully Connected Layer
    model.add(Dense(4096))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # 3rd Fully Connected Layer
    model.add(Dense(1000))
    model.add(Activation('relu'))
    # Add Dropout
    model.add(Dropout(0.4))

    # Output Layer
    model.add(Dense(17))
    model.add(Activation('softmax'))
    return model

def load_model_from_h5():
    print("Please wait, this proccess will take about a min, depending on your machine config")
    model = load_model('model.h5')
    return model


#main

model_training()