import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time

from mltools import *
from sklearn.preprocessing import StandardScaler

def add_bias(datax):
    """ Fonction permettant d'ajouter un biais aux données.
        @param xtrain: float array x array, données auxquelle ajouter un biais
    """
    bias = np.ones((len(datax), 1))
    return np.hstack((bias, datax))

def load_usps(filename):
    """ Fonction de chargement des données.
        @param filename: str, chemin vers le fichier à lire
        @return datax: float array x array, données
        @return datay: float array, labels
    """
    with open(filename, "r") as f:
        f.readline()
        data = [[float(x) for x in l.split()] for l in f if len(l.split())>2]
    tmp=np.array(data)
    return tmp[:,1:],tmp[:,0].astype(int)


def get_usps(l, datax, datay):
    """ Fonction permettant de ne garder que 2 classes dans datax et datay.
        @param l: list(int), liste contenant les 2 classes à garder
        @param datax: float array x array, données
        @param datay: float array, labels
        @param datax_new: float array x array, données pour 2 classes
        @param datay_new: float array, labels pour 2 classes
    """
    if type(l)!=list:
        resx = datax[datay==l,:]
        resy = datay[datay==l]
        return resx,resy

    tmp =   list(zip(*[get_usps(i,datax,datay) for i in l]))
    datax_new, datay_new = np.vstack(tmp[0]),np.hstack(tmp[1])

    return datax_new, datay_new


def show_usps(datax):
    """ Fonction d'affichage des données.
    """
    # plt.title("Prediction : {}".format(datay))
    plt.imshow(datax.reshape((16,16)),interpolation="nearest",cmap="magma")


def plot(datax, datay, model, name=''):
    """ Fonction d'affichage des données gen_arti et de la frontière de
        décision.
    """
    plot_frontiere(datax,lambda x : model.predict(x),step=100)
    plot_data(datax,datay.reshape(1,-1)[0])
    plt.title(name)
    plt.show()

def load_data(classes=10):
    
    
    # Chargement des données USPS
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    # Création des données d'apprentissage et des données d'entraînement
    xtrain, ytrain = get_usps([i for i in range(classes)], alltrainx, alltrainy)
    xtest, ytest = get_usps([i for i in range(classes)], alltestx, alltesty)
    
    #bruit
    bruit = np.random.rand(xtest.shape[0], xtest.shape[1])
    #xtrain = np.where(xtrain+bruit <= 2, xtrain+bruit, xtrain)
    xtest = np.where(xtest+bruit <= 2, xtest+bruit, xtest)
    
    # Normalisation
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    scaler = StandardScaler()
    xtest = scaler.fit_transform(xtest)
    
    # One-Hot Encoding
    onehot = np.zeros((ytrain.size,classes))
    onehot[ np.arange(ytrain.size), ytrain ] = 1
    ytrain = onehot
    
    return xtrain, ytrain, xtest, ytest