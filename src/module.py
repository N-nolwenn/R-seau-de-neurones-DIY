import numpy as np

class Loss(object):
    """ Classe abstraite pour le calcul du coût.
        Note: y et yhat sont des matrices de taille batch × d : chaque
             supervision peut être un vecteur de taille d, pas seulement un
             scalaire comme dans le cas de la régression univariée.
    """
    def forward(self, y, yhat):
        """ Calcule le coût en fonction des deux entrées.
            @param y: (float) array x array, supervision
            @param yhat: (float) array x array, prédiction
            @return : (float) array, vecteur de dimension batch (le nombre d'
                       exemples).
        """
        pass

    def backward(self, y, yhat):
        """ Calcule le gradient du coût par rapport à yhat.
            @param y: (float) array x array, supervision
            @param yhat: (float) array x array, prédiction
        """
        pass
    
class Module(object):
    """ Classe abstraite représentant un module générique du réseau de
        neurones. Ses attributs sont les suivants:
            * self._parameters: obj, stocke les paramètres du module, lorsqu'il
            y en a (ex: matrice de poids pour un module linéaire)
            * self._gradient: obj, permet d'accumuler le gradient calculé
    """
    def __init__(self):
        """ Constructeur de la classe Module.
        """
        self._parameters = None
        self._gradient = None

    def zero_grad(self):
        """ Réinitialise le gradient à 0.
        """
        pass

    def forward(self, X):
        """ Phase passe forward: calcule les sorties du module pour les entrées
            passées en paramètre.
        """
        pass

    def update_parameters(self, gradient_step=1e-3):
        """ Calcule la mise à jour des paramètres du module selon le gradient
            accumulé jusqu'à l'appel à la fonction, avec un pas gradient_step.
        """
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        """ Met a jour la valeur du gradient: calcule le gradient du coût par
            rapport aux paramètres et l'additionne à la variable self._gradient
            en fonction de l'entrée input et des δ de la couche suivante (delta).
            @param input
            @param delta
        """
        pass

    def backward_delta(self, input, delta):
        """ Calcul la derivee de l'erreur: gradient du coût par rapport aux
            entrées en fonction de l'entrée input et des δ de la couche
            suivante (delta).
        """
        pass