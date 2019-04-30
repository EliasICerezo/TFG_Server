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
from keras.applications.mobilenet import MobileNet
import os
from glob import glob
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from numpy import argmax
import time
from keras.optimizers import Adamax, Adam, RMSprop
import matplotlib.pyplot as plt
from keras.applications.vgg16 import VGG16
from keras.engine.saving import load_model
from matplotlib import pyplot as plt
from sklearn.metrics import confusion_matrix
import webbrowser
from keras.losses import mean_squared_error

reducedcsvpath= 'D:\OneDrive\TFG\TFG_Python\HAMReduced.csv'
augmentedcsvpath = 'D:\OneDrive\TFG\TFG_Python\HAMAugmented.csv'
csvpath = 'D:\OneDrive\TFG\TFG_Python\HAM10000_metadata.csv'
bincsvpath = "D:\OneDrive\TFG\TFG_Python\dfrecortado_corregido.csv"
impath = 'D:\OneDrive\TFG\TFG_Python\Images'
h5path = 'D:\OneDrive\TFG\TFG_Python\core\model.h5'
# Constante que da las dimensiones para las distintas imagenes
HEIGHT = WIDTH = 224
# Aqui se ponen los ratios por cada uno de los tipos de imagenes, para ver cuantas modificaciones hay que hacer por cada una de ellas.
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
    'mel': 'Melanoma',
    'bkl': 'Benign keratosis-like lesions ',
    'bcc': 'Basal cell carcinoma',
    'akiec': 'Actinic keratoses',
    'vasc': 'Vascular lesions',
    'df': 'Dermatofibroma'
}

model = None
batch_size=32

def model_training(onlytest = False):
    start = time.time()
    # Local funcions definitions
    def test_gen():
        imglist = np.zeros((len(test_img_path), HEIGHT,WIDTH,3))
        labellist = np.zeros((len(test_img_path), len(labelnum)))
        for i, imgpath in enumerate(test_img_path):
            img = read_img(imgpath)
            label = test_label[i]
            if existsMods:
                mod = test_mods[i]
                if mod.item != 0:
                    plt.imshow(img)
                    plt.show()
                    g_x = image_gen.flow(np.expand_dims(img, axis=0), batch_size=1)
                    x = next(g_x)
                    img = x
                    plt.imshow(img)
                    plt.show()
            imglist[i] = img
            labellist[i] = label
        return (imglist, labellist)

    def valid_gen():
        imglist = np.zeros((len(valid_img_path), HEIGHT, WIDTH, 3))
        labellist = np.zeros((len(valid_img_path), len(labelnum)))
        for i, imgpath in enumerate(valid_img_path):
            try:
                if i % 100 ==0:
                    print("Valid gen: Img leidas= "+str(i))
                img = read_img(imgpath)
                label = valid_label[i]
                if existsMods:
                    mod = valid_mods[i]
                    if mod.item != 0:
                        g_x = image_gen.flow(np.expand_dims(img, axis=0), batch_size=1)
                        x = next(g_x)
                        img = x
                imglist[i] = img
                labellist[i] = label
            except:
                pass
        return (imglist, labellist)

    def train_gen_v2(batch_size=32):
        while(True):
            imglist = np.zeros((batch_size, HEIGHT, WIDTH, 3))
            labellist = np.zeros((batch_size, len(labelnum)))
            for i in range(batch_size):
                rndid = random.randint(0, len(train_img_path) - 1)
                imgpath = train_img_path[train_img_path.index[rndid]]
                img = read_img(imgpath)
                label = train_label[rndid]
                if existsMods:
                    mod = train_mods[rndid]
                    if mod.item != 0:
                        g_x = image_gen.flow(np.expand_dims(img,axis=0), batch_size=1)
                        x= next(g_x)
                        img = x
                imglist[i] = img
                labellist[i] = label
            # print(imglist,labellist)
            yield (imglist, labellist)

    #End of local funcion definitions

    image_gen = ImageDataGenerator(rotation_range=90,
                                   width_shift_range=0.25,
                                   height_shift_range=0.25,
                                   shear_range=0.01,
                                   zoom_range=[0.9, 1.50],
                                   horizontal_flip=True,
                                   vertical_flip=True,
                                   fill_mode='reflect',
                                   data_format='channels_last',
                                   brightness_range=[0.5, 1.75])

    existsMods = True
    isBinary = False
    modsdf = None
    train_df = pd.read_csv(bincsvpath)
    encoder = LabelEncoder()  # Lo usaremos para las labels

    #asignamos este array a la variable labels de cara a usarlo en el testing
    mylist = train_df["tipo"].values
    mylist = list(dict.fromkeys(mylist))
    labels = mylist


    if "tipo" in train_df.columns:
        isBinary=True

    print("ISBINARY: "+str(isBinary))
    if not isBinary:
        train_df['path'] = train_df['image_id'].map(imageid_path_dict.get)
        labelnum = train_df.groupby('dx').size()
        encoder.fit(train_df['dx'])
        labeldf = encoder.transform(train_df['dx'])
        print(labelnum)
        LABELNUM=7
    else:
        labelnum = train_df.groupby("tipo").size()
        print(labelnum)
        encoder.fit(mylist)
        #encoder.fit(train_df["tipo"].astype(str))
        labeldf = encoder.transform(train_df["tipo"])
        LABELNUM=3

    try:
        modsdf = train_df['mods']
        modsdf = modsdf.as_matrix(columns=None)
    except KeyError:
        existsMods = False
    labeldf = to_categorical(labeldf)
    if existsMods:
        print(type(modsdf))
        train_target, test_img_path, train_label, test_label, target_mods, test_mods = train_test_split(
            train_df['path'], labeldf, modsdf, test_size=0.2)
        train_img_path, valid_img_path, train_label, valid_label, train_mods, valid_mods = train_test_split(
            train_target, train_label, target_mods, test_size=0.2)
    else:
        train_target, test_img_path, train_label, test_label = train_test_split(train_df['path'], labeldf,
                                                                                test_size=0.2)
        train_img_path, valid_img_path, train_label, valid_label = train_test_split(train_target, train_label,
                                                                                    test_size=0.2)



    if not onlytest:
        validdata = valid_gen()
        print("Creando modelo y compilandolo")
        model = model_inception_V3()
        model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer=RMSprop())
        callbacks = [
            # ModelCheckpoint('msqe.h5',monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        print('Se comienza el entrenamiento del modelo')
        print(model.metrics_names)
        model.fit_generator(train_gen_v2(), epochs=40, steps_per_epoch=30, verbose=2, validation_data=validdata, callbacks= callbacks)
        print("Entrenamiento completado, se procede al test final")
        test_model(model, test_gen, labels)
    else:
        model = model_load_from_h5(None)
        test_model(model, test_gen, labels)

    finish = time.time()
    print("El entrenamiento ha llevado : "+str(finish-start))


def test_model(model, test_gen, labels):
    test_x, test_y = test_gen()
    result = model.evaluate(test_x, test_y, verbose=1)
    print(model.metrics_names)
    print(result)
    print("Ahora vamos a dibujar la matriz de confusion")
    predictions = model.predict(test_x)
    predictions = [argmax(a) for a in predictions]
    encoder = LabelEncoder()
    encoder.fit([0,1,2])
    predictions = encoder.transform(predictions)
    test_y = [argmax(a) for a in test_y]
    test_y = encoder.transform(test_y)
    plot_confusion_matrix(test_y, predictions, labels)
    webbrowser.open("https://www.youtube.com/watch?v=t6wjCcWC2aE",new=2)




def read_img(imgpath):
    img = cv2.imread(imgpath)
    if img is None:
        print(imgpath)
    img = cv2.resize(img, dsize=(WIDTH, HEIGHT))
    return img




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



def model_load_from_h5(model):
    print("Please wait, this proccess will take about a min, depending on your machine config")
    if model is None:
        print("Loading model.h5")
        model = load_model('D:\OneDrive\TFG\TFG_Python\core\equilibrado.h5')
    return model



def reduce_csv(path):
    curr_df = pd.read_csv(path)
    print(curr_df)
    try:
        print(curr_df['reduced'])
        print("Este csv ya esta reducido")
    except KeyError:
        print("Vamos a proceder a la reduccion del csv")
        i = 0
        curr_df['reduced'] = 1
        for index, row in curr_df.iterrows():
            if index % 100 == 0:
                print("Lineas procesadas: " + str(index))
            if row['dx'] == 'nv':
                i += 1
                if i > 1100:
                    curr_df = curr_df.drop(curr_df.index[curr_df.index.get_loc(row.name)])
        print(curr_df)
        print("Reduccion terminada")
        curr_df.to_csv(reducedcsvpath)


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
        curr_df.to_csv(reducedcsvpath)


def predict_image(imagepath, mod=None):
    #TODO try except y ver posibles errores
        train_df = pd.read_csv(bincsvpath)
        encoder = LabelEncoder()  # Lo usaremos para las labels
        encoder.fit(train_df['tipo'])
        model = model_load_from_h5(mod)
        img = read_img(imagepath)
        img = np.expand_dims(img, axis=0)
        result = model.predict(img, verbose=0)
        result = np.round_(result)
        print(result[0])
        decoded = encoder.inverse_transform([argmax(result)])
        print(decoded)
        return decoded, model


def plot_confusion_matrix(y_true, y_pred, classes,
                          normalize=True,
                          title=None,
                          cmap=plt.cm.Blues):
    """
    This function prints and plots the confusion matrix.
    Normalization can be applied by setting `normalize=True`.
    """
    print(classes)
    if not title:
        if normalize:
            title = 'Normalized confusion matrix'
        else:
            title = 'Confusion matrix, without normalization'

    # Compute confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    # Only use the labels that appear in the data
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        print("Normalized confusion matrix")
    else:
        print('Confusion matrix, without normalization')

    print(cm)

    fig, ax = plt.subplots()
    im = ax.imshow(cm, interpolation='nearest', cmap=cmap)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=classes, yticklabels=classes,
           title=title,
           ylabel='True label',
           xlabel='Predicted label')

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="right",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f' if normalize else 'd'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    plt.show()
    return ax

if __name__ == '__main__':
    model_training()