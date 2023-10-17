from cencapsulage import *

class SoftMax (Module) :
    """ Module représentant une couche de transformation SoftMax.
    """
    def __init__(self):
        """ Constructeur du module SoftMax.
        """
        super().__init__()

    def forward(self, X):
        """ Phase passe forward: calcule les sorties du module pour les entrées
            passées en paramètre.
            Elle prend en entrée une matrice de taille batch * input et rend en
            sortie une matrice de taille batch * input.
            @param X: (float) array x array, matrice des entrées (batch x input)
        """
        e_x = np.exp(X)
        return e_x / e_x.sum( axis = 1 ).reshape(-1,1)

    def backward_delta(self, X, delta):
        """ Calcul la dérivée de l'erreur du module Sigmoide par rapport aux
            δ de la couche suivante (delta).
        """
        s = self.forward( np.array(X) )
        return s * ( 1 - s ) * delta

    def update_parameters(self, gradient_step=1e-3):
        """ Calcule la mise à jour des paramètres du module selon le gradient
            accumulé jusqu'à l'appel à la fonction, avec un pas gradient_step.
        """
        pass

class CE(Loss):
    """ Classe pour la fonction de coût cross-entropique.
    """
    def forward (self, y, yhat) :
        """ Calcul du coût cross-entropique.
            @return : (float) array, coût de taille batch.
        """
        return - np.sum( y * yhat , axis = 1 )

    def backward (self, y, yhat) :
        """ Calcule le gradient du coût cross-entropique par rapport à yhat.
            @return : (float) array x array, gradient de taille batch x d.
        """
        return - y

class CESM(Loss):
    """ Classe pour la fonction de coût cross-entropique appliqué au log SoftMax.
    """
    def forward (self, y, yhat) :
        """ Calcul du coût cross-entropique appliqué au log SoftMax.
            @return : (float) array, coût de taille batch.
        """
        return - np.sum( y * yhat , axis = 1 ) + np.log( np.sum( np.exp(yhat), axis = 1 ) )

    def backward (self, y, yhat) :
        """ Calcule le gradient du coût cross-entropique par rapport à yhat.
            @return : (float) array x array, gradient de taille batch x d.
        """
        s = SoftMax().forward( yhat )
        return - y + s * ( 1 - s )
    
def test_multi(neuron=10, niter=300, gradient_step=1e-3, batch_size=None):
    # Chargement des données USPS
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    # Création des données d'apprentissage et des données d'entraînement
    classes = 10
    xtrain, ytrain = get_usps([i for i in range(classes)], alltrainx, alltrainy)
    xtest, ytest = get_usps([i for i in range(classes)], alltestx, alltesty)
    
    # Normalisation
    scaler = StandardScaler()
    xtrain = scaler.fit_transform(xtrain)
    scaler = StandardScaler()
    xtest = scaler.fit_transform(xtest)
    
    # One-Hot Encoding
    onehot = np.zeros((ytrain.size,classes))
    onehot[ np.arange(ytrain.size), ytrain ] = 1
    ytrain = onehot
    
    # Initialisation batch_size
    if batch_size == None:
        batch_size = len(xtrain)
        
    # Récupération des tailles des entrées
    batch, output = ytrain.shape
    batch, input = xtrain.shape
    
    # Initialisation des couches du réseau et de la loss
    ce = CESM()
    linear_1 = Linear(input+1, neuron)
    tanh = TanH()
    linear_2 = Linear(neuron, output)
    softmax = SoftMax()
    
    # Liste des couches du réseau de neurones
    net = [linear_1, tanh, linear_2, softmax]
    
    # Création et test sur un réseau de neurones non linéaire
    
    time_start = time.time()
    
    nonlin = NonLineaireSeq()
    nonlin.SGD(net, ce, xtrain, ytrain, batch_size=batch_size, niter=niter, gradient_step=gradient_step)
    #nonlin.fit(xtrain, ytrain, batch_size=len(ytrain), niter=100, gradient_step=0.01)
    
    # Evolution de la loss
    plt.figure()
    plt.title('Cross-Entropie Softmax loss avec {} neurones, niter = {}, gradient_step = {}'.format(neuron, niter, gradient_step))
    plt.plot(nonlin.list_loss, label='loss', c='midnightblue')
    plt.legend()
    plt.xlabel('Nombre d\'itérations')
    
    # Test sur les données d'apprentissage
    ypred = nonlin.predict(xtest)
    
    print("\nTemps d'exécution: ", time.time() - time_start )
    print("Score de bonne classification: ", np.mean( ypred == ytest ))
    
    return xtest, ypred