import cv2
import pandas as pd
from keras.engine.saving import load_model
from matplotlib import pyplot as plt
from numpy import argmax
from sklearn.metrics import confusion_matrix
from sklearn.preprocessing import LabelEncoder
import numpy as np

reducedcsvpath= 'D:\OneDrive\TFG\TFG_Python\HAMReduced.csv'
bincsvpath = "D:\OneDrive\TFG\TFG_Python\dfrecortado_corregido.csv"
HEIGHT = WIDTH = 224
# Aqui se ponen los ratios por cada uno de los tipos de imagenes, para ver cuantas modificaciones hay que hacer por cada una de ellas.
AKIEC = 19
BCC = 11
BKL = 5
DF = 55
MEL = 4
VASC = 44

# Carga un modelo desde una ruta concreta. En caso de que reciba un modelo devuelve el mismo
def model_load_from_h5(model):
    print("Please wait, this proccess will take about a min, depending on your machine config")
    if model is None:
        print("Loading model.h5")
        model = load_model('D:\OneDrive\TFG\TFG_Python\models\inceptionresnetv2\model06dropout.h5')
    return model

# Funcion que opera un csv y realiza el submuestreo
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

# Funcion que realiza el submuestreo del csv de HAM10000
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

# Funcion que realiza las predicciones de una imagen recibida
def predict_image(imagepath, mod=None):
    train_df = pd.read_csv(bincsvpath)
    encoder = LabelEncoder()  # Lo usaremos para las labels
    encoder.fit(train_df['tipo'])
    model = model_load_from_h5(mod)
    img = read_image(imagepath) #Leemos la imagen
    img = np.expand_dims(img, axis=0)
    result = model.predict(img, verbose=0) #Se realiza la prediccion
    result = np.round_(result)
    decoded = encoder.inverse_transform([argmax(result)]) #Decodificamos el resultado que viene en formato to_categorical
    return decoded, model

#Funcion que pinta la matriz de confusion
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


def read_image(imgpath):
    img = cv2.imread(imgpath)
    if img is None:
        print(imgpath)
    img = cv2.resize(img, dsize=(WIDTH, HEIGHT))
    return img
