import random

from keras.layers import *
from keras.callbacks import ReduceLROnPlateau,EarlyStopping
from sklearn.model_selection import train_test_split
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import os
from glob import glob
from keras.utils import to_categorical
from keras.preprocessing.image import ImageDataGenerator
from numpy import argmax
import time
from keras.optimizers import RMSprop
from matplotlib import pyplot as plt
import webbrowser
from core.modelstructures import model_inception_V3, model_inception_resnetV2, model_VGG16, model_mobilenet

reducedcsvpath= 'D:\OneDrive\TFG\TFG_Python\HAMReduced.csv'
augmentedcsvpath = 'D:\OneDrive\TFG\TFG_Python\HAMAugmented.csv'
csvpath = 'D:\OneDrive\TFG\TFG_Python\HAM10000_metadata.csv'
bincsvpath = "D:\OneDrive\TFG\TFG_Python\dfrecortado_corregido.csv"
impath = 'D:\OneDrive\TFG\TFG_Python\Images'
h5path = 'D:\OneDrive\TFG\TFG_Python\core\model.h5'
# Constante que da las dimensiones para las distintas imagenes
HEIGHT = WIDTH = 224



imageid_path_dict = {os.path.splitext(os.path.basename(x))[0]: x
                     for x in glob(os.path.join(impath, '*.jpg'))}


from core.modelfunctions import model_load_from_h5, plot_confusion_matrix, read_image

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

"""
    Esta función realiza el entrenamiento del modelo
    Recibe como argumentos:
    onlytest, que en caso de que sea True hará las pruebas del modelo que esté referenciado en load_model_from_h5
"""


def model_training(onlytest = False):
    start = time.time()
    #Definicion de los generadores locales
    def test_gen():
        imglist = np.zeros((len(test_img_path), HEIGHT,WIDTH,3))
        labellist = np.zeros((len(test_img_path), len(labelnum)))
        for i, imgpath in enumerate(test_img_path):
            img = read_image(imgpath)
            label = test_label[i]
            # En caso de que se pase un csv con data augmentation se realizan las modificaciones pertinentes
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

    #Generador para la los datos de validación al final de cada epoca
    def valid_gen():
        # Se crea un tensor, se llena de imágenes y se devuelven
        imglist = np.zeros((len(valid_img_path), HEIGHT, WIDTH, 3))
        labellist = np.zeros((len(valid_img_path), len(labelnum)))
        for i, imgpath in enumerate(valid_img_path):
            try:
                if i % 100 ==0:
                    print("Valid gen: Img leidas= "+str(i))
                img = read_image(imgpath)
                label = valid_label[i]
                # En caso de que se pase un csv con data augmentation se realizan las modificaciones pertinentes
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

    #Generador de entrenamiento
    def train_gen_v2(batch_size=32):
        while(True):
            # Se crea un tensor del batch size indicado
            imglist = np.zeros((batch_size, HEIGHT, WIDTH, 3))
            labellist = np.zeros((batch_size, len(labelnum)))
            # Se rellena de imágenes de manera aleatoria
            for i in range(batch_size):
                rndid = random.randint(0, len(train_img_path) - 1)
                imgpath = train_img_path[train_img_path.index[rndid]]
                img = read_image(imgpath)
                label = train_label[rndid]
                # En caso de que se pase un csv con data augmentation se realiza el data augmentation correspondiente
                if existsMods:
                    mod = train_mods[rndid]
                    if mod.item != 0:
                        g_x = image_gen.flow(np.expand_dims(img,axis=0), batch_size=1)
                        x= next(g_x)
                        img = x
                imglist[i] = img
                labellist[i] = label
            yield (imglist, labellist)

    #Fin de las definiciones locales

    # Generador de imágenes para los csv con data augmentation
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

    #Variables inherentes a la ejecucion que se modificaran a medida que se vaya comprobando la configuracion
    existsMods = True # Es un dataset con data aumentation
    isBinary = False # La clasificacion es trivaluada
    modsdf = None # Datos de las modificaciones para data augmentation
    train_df = pd.read_csv(bincsvpath)
    encoder = LabelEncoder()  # Lo usaremos para las labels

    #asignamos este array a la variable labels de cara a usarlo en el testing
    mylist = train_df["tipo"].values
    mylist = list(dict.fromkeys(mylist))
    labels = mylist

    #Comprobamos si en el csv que nos pasan existe la columna tipo y por tanto la clasificación es trivaluada
    if "tipo" in train_df.columns:
        isBinary=True
    #Se informa de la clasificacion que se va a hacar
    print("ISBINARY: "+str(isBinary))
    # Se realiza la configuracion en base a si al clasificacion es trivaluada o no
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
        labeldf = encoder.transform(train_df["tipo"])
        LABELNUM=3

    #Comprobamos si el csv esta configurado para data augmentation
    try:
        modsdf = train_df['mods']
        modsdf = modsdf.as_matrix(columns=None)
    except KeyError:
        existsMods = False
    labeldf = to_categorical(labeldf)
    # Hacemos la particion del dataset
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
        #Seleccionamos y compilamos el modelo el modelo
        print("Creando modelo y compilandolo")
        model = model_inception_V3()
        model.compile(metrics=['accuracy'], loss='mean_squared_error', optimizer=RMSprop())
        #Añadimos los callbacks
        callbacks = [
            # ModelCheckpoint('msqe.h5',monitor='val_loss', save_best_only=True, verbose=1),
            ReduceLROnPlateau(monitor='val_loss', patience=5, verbose=1),
            EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
        ]
        print('Se comienza el entrenamiento del modelo')
        print(model.metrics_names)
        #Entrenamos el modelo
        model.fit_generator(train_gen_v2(), epochs=1, steps_per_epoch=1, verbose=2, validation_data=validdata, callbacks= callbacks)
        print("Entrenamiento completado, se procede al test final")
        #Probamos el modelo
        test_model(model, test_gen, labels)
    else:
        model = model_load_from_h5(None)
        test_model(model, test_gen, labels)

    finish = time.time()
    print("El entrenamiento ha llevado : "+str(finish-start))


""" 
    En esta funcion se prueba el modelo, recibe como argumentos:
    El modelo a probar
    El generador de tests
    Los valores reales de clasificación de los datos que se van a probar
"""
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


if __name__ == '__main__':
    model_training()


