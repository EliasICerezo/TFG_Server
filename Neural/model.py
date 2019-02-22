import random
from keras import Model
from keras.layers import *
from keras.callbacks import ModelCheckpoint,ReduceLROnPlateau
from sklearn.model_selection import train_test_split
import cv2
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from keras.applications.inception_resnet_v2 import InceptionResNetV2
import os
from glob import glob
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator

csvpath = 'D:\OneDrive\TFG\TFG_Python\HAM10000_metadata.csv'
impath = 'D:\OneDrive\TFG\TFG_Python\Images'
h5path = 'D:\OneDrive\TFG\TFG_Python\Neural\model.h5'
#Constante que da las dimensiones para las distintas imagenes
HEIGHT = WIDTH = 185
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
    def valid_gen():
        imglist = np.zeros((len(valid_img_path), HEIGHT, WIDTH, 3))
        labellist = np.zeros((len(valid_img_path), len(labelnum)))
        for i, imgpath in enumerate(valid_img_path):
            img = read_img(imgpath)
            label = valid_label[i]
            imglist[i] = img
            labellist[i] = label
        return (imglist, labellist)

    def train_gen(batch_size=64):
        while (True):
            imglist = np.zeros((batch_size, HEIGHT, WIDTH, 3))
            labellist = np.zeros((batch_size, len(labelnum)))
            for i in range(batch_size):
                rndid = random.randint(0, len(train_img_path) - 1)
                imgpath = train_img_path[train_img_path.index[rndid]]
                img = read_img(imgpath)
                label = train_label[rndid]
                imglist[i] = img
                labellist[i] = label
            yield (imglist, labellist)

    #End of local funcion definitions

    train_df = pd.read_csv(csvpath)
    train_df['path'] = train_df['image_id'].map(imageid_path_dict.get)

    labelnum = train_df.groupby('dx').size()

    encoder = LabelEncoder()  # Lo usaremos para las labels
    encoder.fit(train_df['dx'])
    print(encoder.classes_)
    labeldf = encoder.transform(train_df['dx'])
    labeldf = to_categorical(labeldf)
    train_img_path, valid_img_path, train_label, valid_label = train_test_split(train_df['path'], labeldf, test_size=0.2)

    validdata = valid_gen()


    print("Creando modelo y compilandolo")
    model = create_model()
    model.compile(metrics=['accuracy'], loss='categorical_crossentropy', optimizer='Adam')
    callbacks = [
        ModelCheckpoint('model.h5',monitor='val_loss', save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', patience=5)
    ]

    model.fit_generator(train_gen(), epochs=50, steps_per_epoch=100, verbose=1, validation_data=validdata, callbacks= callbacks)






def read_img(imgpath):
    img = cv2.imread(imgpath)
    img = cv2.resize(img, dsize=(WIDTH, HEIGHT))
    return img



def create_model():
    model = InceptionResNetV2(include_top = False, input_shape = (HEIGHT, WIDTH, 3), weights='imagenet')
    addition = GlobalAveragePooling2D()(model.output)
    addition = Dropout(0.5)(addition)
    addition = Dense(256, activation='relu')(addition)
    addition = Dense(LABELNUM, activation='softmax')(addition)
    model = Model(model.inputs, addition)
    return model




#main
model_training()