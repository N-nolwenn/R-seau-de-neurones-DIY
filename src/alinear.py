from module import *

class MSELoss(Loss):
    """ Classe pour la fonction de coût aux moindres carrés.
    """
    def forward(self, y, yhat):
        """ Calcul du coût aux moindres carrés (mse).
            @return : (float) array, coût de taille batch.
        """
        assert y.shape == yhat.shape
        return np.linalg.norm( y-yhat, axis=1) ** 2

    def backward(self, y, yhat):
        """ Calcule le gradient du coût aux moindres carrés par rapport à yhat.
            @return : (float) array x array, gradient de taille batch x d.
        """
        assert y.shape==yhat.shape
        return -2 * (y-yhat)

class Linear(Module):
    """ Module représentant une couche linéaire avec input entrées et output
        sorties.
            * self._parameters: float array x array, matrice de poids pour la
              couche linéaire, de taille input x output.
    """
    def __init__(self, input, output):
        """ Constructeur du module Linear.
        """
        self.input = input
        self.output = output

        """ Initialisation des poids 
        """
        self._parameters = 2 * ( np.random.rand(self.input, self.output) - 0.5 )
        self.zero_grad()

    def zero_grad(self):
        """ Réinitialise le gradient à 0.
        """
        self._gradient = np.zeros((self.input, self.output))

    def forward(self, X):
        """ Phase passe forward: calcule les sorties du module pour les entrées
            passées en paramètre.
            Elle prend en entrée une matrice de taille batch * input et rend en
            sortie une matrice de taille batch * output.
            @param X: (float) array x array, matrice des entrées (batch x input)
        """
        return np.dot( X, self._parameters)

    def update_parameters(self, gradient_step=1e-3):
        """ Calcule la mise à jour des paramètres du module selon le gradient
            accumulé jusqu'à l'appel à la fonction, avec un pas gradient_step.
        """
        self._parameters -= gradient_step * self._gradient

    def backward_update_gradient(self, input, delta):
        """ Met a jour la valeur du gradient: calcule le gradient du coût par
            rapport aux paramètres et l'additionne à la variable self._gradient
            en fonction de l'entrée input et des δ de la couche suivante (delta).
            Le gradient calculé est de taille input x output
            @param X: (float) array x array, matrice des entrées (batch x input)
            @param delta: (float) array x array, matrice de dimensions
                          batch x output
        """
        self._gradient += np.dot( input.T, delta )

    def backward_delta(self, input, delta):
        """ Calcul la derivee de l'erreur: gradient du coût par rapport aux
            entrées en fonction de l'entrée input et des δ de la couche
            suivante (delta).
        """
        return np.dot( delta, self._parameters.T )

