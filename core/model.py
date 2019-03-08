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
from numpy import argmax
import time

newcsvpath= 'D:\OneDrive\TFG\TFG_Python\HAMNewCsv.csv'
csvpath = 'D:\OneDrive\TFG\TFG_Python\HAM10000_metadata.csv'
impath = 'D:\OneDrive\TFG\TFG_Python\Images'
h5path = 'D:\OneDrive\TFG\TFG_Python\core\model.h5'
#Constante que da las dimensiones para las distintas imagenes
HEIGHT = WIDTH = 224
LABELNUM = 7
#Aqui se ponen los ratios por cada uno de los tipos de imagenes, para ver cuantas modificaciones hay que hacer por cada una de ellas.
AKIEC = 19
BCC = 11
BKL = 5
DF = 55
MEL = 4
VASC = 44

imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(impath, '*.jpg'))}

lesion_type_dict = {
    'nv': 'Melanocytic nevi',
    'mel': 'Dermatofibroma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

def model_training():
    start = time.time()
    #Local funcions definitions
    def test_gen():
        imglist = np.zeros((len(test_img_path), HEIGHT,WIDTH,3))
        labellist = np.zeros((len(test_img_path), len(labelnum)))
        for i, imgpath in enumerate(test_img_path):
            img = read_img(imgpath)
            label = test_label[i]
            mod = test_mods[i]
            if mod.item != 0:
                g_x = image_gen.flow(np.expand_dims(img, axis=0), batch_size=1)
                x = next(g_x)
                img = x
            imglist[i] = img
            labellist[i] = label
        return (imglist, labellist)

    def valid_gen():
        imglist = np.zeros((len(valid_img_path), HEIGHT, WIDTH, 3))
        labellist = np.zeros((len(valid_img_path), len(labelnum)))
        for i, imgpath in enumerate(valid_img_path):
            img = read_img(imgpath)
            label = valid_label[i]
            mod = valid_mods[i]
            if mod.item != 0:
                g_x = image_gen.flow(np.expand_dims(img, axis=0), batch_size=1)
                x = next(g_x)
                img = x
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

    def train_gen_v2(batch_size=32):
        while(True):
            imglist = np.zeros((batch_size, HEIGHT, WIDTH, 3))
            labellist = np.zeros((batch_size, len(labelnum)))
            for i in range(batch_size):
                rndid = random.randint(0, len(train_img_path) - 1)
                imgpath = train_img_path[train_img_path.index[rndid]]
                img = read_img(imgpath)
                label = train_label[rndid]
                mod = train_mods[rndid]
                if mod.item != 0:
                    g_x = image_gen.flow(np.expand_dims(img,axis=0), batch_size=1)
                    x= next(g_x)
                    img = x
                imglist[i] = img
                labellist[i] = label
            yield (imglist, labellist)

    #End of local funcion definitions

    image_gen = ImageDataGenerator(rotation_range=15,
                                   width_shift_range=0.1,
                                   height_shift_range=0.1,
                                   shear_range=0.01,
                                   zoom_range=[0.9, 1.25],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='reflect',
                                   data_format='channels_last',
                                   brightness_range=[0.5, 1.5])

    train_df = pd.read_csv(newcsvpath)
    train_df['path'] = train_df['image_id'].map(imageid_path_dict.get)

    labelnum = train_df.groupby('dx').size()

    encoder = LabelEncoder()  # Lo usaremos para las labels
    encoder.fit(train_df['dx'])
    labeldf = encoder.transform(train_df['dx'])
    modsdf = train_df['mods']
    modsdf = modsdf.as_matrix(columns=None)
    print_samples(encoder, labeldf)
    labeldf = to_categorical(labeldf)
    print(type(labeldf))
    print(type(modsdf))
    train_target, test_img_path, train_label, test_label, target_mods, test_mods = train_test_split(train_df['path'], labeldf, modsdf, test_size=0.2)
    train_img_path, valid_img_path, train_label, valid_label, train_mods, valid_mods = train_test_split(train_target, train_label, target_mods, test_size=0.2)
    validdata = valid_gen()
    print("Mods len: "+str(len(train_mods)))
    print("Labels len: "+str(len(train_label)))

    print("Creando modelo y compilandolo")
    model = model_inception_resnetV2()
    model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='Adam')
    callbacks = [
        ModelCheckpoint('model.h5',monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
        EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    ]
    print('Se comienza el entrenamiento del modelo')
    print(model.metrics_names)
    model.fit_generator(train_gen_v2(), epochs=80, steps_per_epoch=80, verbose=2, validation_data=validdata, callbacks= callbacks)
    print("Entrenamiento completado, se procede al test final")
    test_x, test_y = test_gen()
    result = model.evaluate(test_x, test_y, verbose=1)
    print(model.metrics_names)
    print(result)
    finish = time.time()
    print("El entrenamiento ha llevado : "+str(finish-start))


def print_samples(encoder, labeldf):
    print("Number of samples in the entire dataset: ")
    print(encoder.inverse_transform([0]) + " : " + str(list(labeldf).count(0)))
    print(encoder.inverse_transform([1]) + " : " + str(list(labeldf).count(1)))
    print(encoder.inverse_transform([2]) + " : " + str(list(labeldf).count(2)))
    print(encoder.inverse_transform([3]) + " : " + str(list(labeldf).count(3)))
    print(encoder.inverse_transform([4]) + " : " + str(list(labeldf).count(4)))
    print(encoder.inverse_transform([5]) + " : " + str(list(labeldf).count(5)))
    print(encoder.inverse_transform([6]) + " : " + str(list(labeldf).count(6)))


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

def augmentate_csv(path):
    curr_df = pd.read_csv(path)
    try:
        print(curr_df['mods'])
        print("El csv esta ensanchado")
    except KeyError:
        print("Se procede al ensanchamiento del csv")
        curr_df['mods'] = 0
        for index, row in curr_df.iterrows():
            if index %100 ==0:
                print("Lineas procesadas: "+str(index))
            if row['dx'] == 'akiec' and row['mods']==0:
                for i in range(1, (AKIEC+1) ):
                    new_row = row
                    new_row['mods'] = i
                    curr_df = curr_df.append(new_row, ignore_index=True)
            elif row['dx'] == 'bcc' and row['mods']==0:
                for i in range(1, (BCC + 1)):
                    new_row = row
                    new_row['mods'] = i
                    curr_df = curr_df.append(new_row, ignore_index=True)
            elif row['dx'] == 'bkl' and row['mods']==0:
                for i in range(1, (BKL + 1)):
                    new_row = row
                    new_row['mods'] = i
                    curr_df = curr_df.append(new_row, ignore_index=True)
            elif row['dx'] == 'df' and row['mods']==0:
                for i in range(1, (DF + 1)):
                    new_row = row
                    new_row['mods'] = i
                    curr_df = curr_df.append(new_row, ignore_index=True)
            elif row['dx'] == 'mel' and row['mods']==0:
                for i in range(1, (MEL + 1)):
                    new_row = row
                    new_row['mods'] = i
                    curr_df = curr_df.append(new_row, ignore_index=True)
            elif row['dx'] == 'vasc' and row['mods']==0:
                for i in range(1, (VASC + 1)):
                    new_row = row
                    new_row['mods'] = i
                    curr_df = curr_df.append(new_row, ignore_index=True)
        print(curr_df)
        print("Ensanchamiento terminado")
        curr_df.to_csv(newcsvpath)


def predict_image(image):
    try:
        load_model_from_h5()
        #hacemos el predict de la imagen, extraemos el label
    except:
        return None

