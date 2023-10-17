from alinear import *

from utils import *
import numpy as np
import matplotlib.pyplot as plt
import random as rd
import time

from mltools import *
from sklearn.preprocessing import StandardScaler

class TanH(Module):
    """ Module représentant une couche de transformation tanh.
    """
    def __init__(self):
        """ Constructeur du module TanH.
        """
        super().__init__()

    def forward(self, X):
        """ Phase passe forward: calcule les sorties du module pour les entrées
            passées en paramètre.
            Elle prend en entrée une matrice de taille batch * input et rend en
            sortie une matrice de taille batch * input.
            @param X: (float) array x array, matrice des entrées (batch x input)
        """
        return np.tanh(X)

    def backward_delta(self, X, delta):
        """ Calcul la dérivée de l'erreur du module TanH par rapport aux
            δ de la couche suivante (delta).
        """
        return ( 1 - np.tanh(X)**2 ) * delta

    def update_parameters(self, gradient_step=1e-3):
        """ Calcule la mise à jour des paramètres du module selon le gradient
            accumulé jusqu'à l'appel à la fonction, avec un pas gradient_step.
        """
        pass


class Sigmoide(Module):
    """ Module représentant une couche de transformation sigmoide.
    """
    def __init__(self):
        """ Constructeur du module Sigmoide.
        """
        super().__init__()

    def forward(self, X):
        """ Phase passe forward: calcule les sorties du module pour les entrées
            passées en paramètre.
            Elle prend en entrée une matrice de taille batch * input et rend en
            sortie une matrice de taille batch * input.
            @param X: (float) array x array, matrice des entrées (batch x input)
        """
        return 1 / (1 + np.exp(-X))

    def backward_delta(self, X, delta):
        """ Calcul la dérivée de l'erreur du module Sigmoide par rapport aux
            δ de la couche suivante (delta).
        """
        #print("SIGMOIDE =",( np.exp(-X) / ( 1 + np.exp(-X) )**2 ))
        #input()
        return ( np.exp(-X) / ( 1 + np.exp(-X) )**2 ) * delta

    def update_parameters(self, gradient_step=1e-3):
        """ Calcule la mise à jour des paramètres du module selon le gradient
            accumulé jusqu'à l'appel à la fonction, avec un pas gradient_step.
        """
        pass
    

class NonLineaire:
    """ Classe pour un classifieur non-linéaire par réseau de neurones.
    """
    def fit(self, xtrain, ytrain, niter=100, gradient_step=1e-5, neuron=100):
        """ Classification non-linéaire sur les données d'apprentissage.
            @param xtrain: float array x array, données d'apprentissage
            @param ytrain: int array, labels sur les données d'apprentissage
            @param niter: int, nombre d'itérations
            @param gradient_step: float, pas de gradient
            @param neuron: nombre de neurones dans une couche
        """
        # Ajout d'un biais aux données
        xtrain = add_bias (xtrain)

        # Récupération des tailles des entrées
        batch, output = ytrain.shape
        batch, input = xtrain.shape

		# Initialisation de la listte des loss
        self.list_loss = []

        # Initialisation des couches du réseau et de la loss
        self.mse = MSELoss()
        self.linear_1 = Linear(input, neuron)
        self.tanh = TanH()
        self.linear_2 = Linear(neuron, output)
        self.sigmoide = Sigmoide()

        for i in range(niter):

            # ETAPE 1: Calcul de l'état du réseau (phase forward)
            res1 = self.linear_1.forward(xtrain)
            res2 = self.tanh.forward(res1)
            res3 = self.linear_2.forward(res2)
            res4 = self.sigmoide.forward(res3)

            # ETAPE 2: Phase backward (rétro-propagation du gradient de la loss
            #          par rapport aux paramètres et aux entrées)
            last_delta = self.mse.backward(ytrain, res4)

            delta_sig = self.sigmoide.backward_delta(res3, last_delta)
            delta_lin = self.linear_2.backward_delta(res2, delta_sig)
            delta_tan = self.tanh.backward_delta(res1, delta_lin)

            self.linear_1.backward_update_gradient(xtrain, delta_tan)
            self.linear_2.backward_update_gradient(res2, delta_sig)
            # ETAPE 3: Mise à jour des paramètres du réseau (matrice de poids w)
            self.linear_1.update_parameters(gradient_step)
            self.linear_2.update_parameters(gradient_step)
            self.linear_1.zero_grad()
            self.linear_2.zero_grad()
            self.list_loss.append(np.mean( self.mse.forward(ytrain, res4)) )

        # Affichage de la loss
        print("\nErreur mse :", np.mean( self.mse.forward(ytrain, res4) ) )

    def predict(self, xtest):
        """ Prédiction sur des données. Il s'agit simplement d'un forward.
        """
        # Ajout d'un biais aux données
        xtest = add_bias (xtest)

        # Phase passe forward
        res = self.linear_1.forward(xtest)
        res = self.tanh.forward(res)
        res = self.linear_2.forward(res)
        res = self.sigmoide.forward(res)

        return np.argmax(res, axis = 1)
	

def test_non_lin(neuron=10, niter=1000, gradient_step=1e-3, batch_size=None):    
    # Création de données artificielles suivant 4 gaussiennes
    datax, datay = gen_arti(epsilon=0.1, data_type=1)
    
    # Descente de gradient batch par défaut
    if batch_size == None:
        batch_size = len(datay)
        
    # Normalisation des données
    scaler = StandardScaler()
    datax = scaler.fit_transform(datax)
    
    # One-Hot Encoding
    datay = np.array([ 0 if d == -1 else 1 for d in datay ])
    onehot = np.zeros((datay.size, 2))
    onehot[ np.arange(datay.size), datay ] = 1
    datay = onehot
    
    # Création et test sur un réseau de neurones non linéaire
    
    time_start = time.time()
    
    batch, output = datay.shape
    batch, input = datax.shape
    
    mse = MSELoss()
    linear_1 = Linear(input+1, neuron)
    tanh = TanH()
    linear_2 = Linear(neuron, output)
    sigmoide = Sigmoide()
    
    net = [linear_1, tanh, linear_2, sigmoide]
    
    nonlin = NonLineaire()
    nonlin.fit(datax, datay, niter=1000, neuron=10, gradient_step=1e-3)
    
    # Test sur les données d'apprentissage
    ypred = nonlin.predict(datax)
    datay = np.argmax(datay, axis=1)
    
    print("\nTemps d'exécution: ", time.time() - time_start )
    print("Score de bonne classification: ", np.mean( ypred == datay ))
    plot(datax, datay, nonlin, name='Regression non Linéaire, n_neurons = {}, niter = {}, gradient_step = {}'.format(neuron, niter, gradient_step))
    
    # Evolution de la loss
    plt.figure()
    plt.title('Evolution de la loss')
    plt.plot(nonlin.list_loss, label='loss', c='midnightblue')
    plt.legend()
    plt.xlabel('Nombre d\'itérations')