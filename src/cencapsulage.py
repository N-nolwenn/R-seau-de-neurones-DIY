from bnonlinear import *


class Sequentiel:
    """ Classe qui permet d'ajouter des modules en série et d'automatiser les
        procédure forward et backward sur toutes les couches.
            * self.modules: list(Module), liste des modules du réseau
            * self.loss: Loss, coût à minimiser
    """
    def __init__(self, modules, loss):
        """ Constructeur de la classe Sequentiel.
        """
        self.modules = modules
        self.loss = loss

    def fit(self, xtrain, ytrain):
        """ Réalise une itération forward et backward sur les couches du
            paramètre self.modules.
            @param xtrain: float array x array, données d'apprentissage
            @param ytrain: int array, labels sur les données d'apprentissage
        """
        # ETAPE 1: Calcul de l'état du réseau (phase forward)
        res_forward = [ self.modules[0].forward(xtrain) ]

        for j in range(1, len(self.modules)):
            res_forward.append( self.modules[j].forward( res_forward[-1] ) )

        # ETAPE 2: Phase backward (rétro-propagation du gradient de la loss
        #          par rapport aux paramètres et aux entrées)
        deltas =  [ self.loss.backward( ytrain, res_forward[-1] ) ]

        for j in range(len(self.modules) - 1, 0, -1):
            deltas += [ self.modules[j].backward_delta( res_forward[j-1], deltas[-1] ) ]

        return res_forward, deltas


class Optim:
    """ Classe qui permet de condenser une itération de gradient. Elle calcule
        la sortie du réseau self.net, exécute la passe backward et met à jour
        les paramètres du réseau.
            * self.net: list(Module), réseau de neurones sous forme d'une liste
                        de Modules correspondant aux différentes couches.
            * self.loss: Loss, coût à minimiser
            * self.eps: float, pas pour la mise-à-jour du gradient
    """
    def __init__(self, net, loss, eps):
        """ Constructeur de la classe Optim.
        """
        self.net = net
        self.loss = loss
        self.eps = eps
        self.sequentiel = Sequentiel(net, loss)

    def step(self, batch_x, batch_y):
        """ Calcule la sortie du réseau, exécute la passe-backward et met à
            jour les paramètres du réseau.
            @param batch_x: float array x array, batch d'apprentissage
            @param batch_y: int array, labels sur le batch d'apprentissage
        """
        # ETAPE 1: Calcul de l'état du réseau (phase forward) et passe backward
        res_forward, deltas = self.sequentiel.fit(batch_x, batch_y)

        # ETAPE 2: Phase backward par rapport aux paramètres et mise-à-jour
        for j in range(len(self.net)):

            # Mise-à-jour du gradient
            if j == 0:
                self.net[j].backward_update_gradient(batch_x, deltas[-1])
            else:
                self.net[j].backward_update_gradient(res_forward[j-1], deltas[-j-1])

            # Mise-à-jour des paramètres
            self.net[j].update_parameters(self.eps)
            self.net[j].zero_grad()

    def predict(self, xtest, onehot=False):
        """ Prédiction sur des données. Il s'agit simplement d'un forward.
        """
        # Phase passe forward
        res_forward = [ self.net[0].forward(xtest) ]

        for j in range(1, len(self.net)):
            res_forward.append( self.net[j].forward( res_forward[-1] ) )
        
        yhat = np.argmax(res_forward[-1], axis=1)
        
        if onehot:
            onehot = np.zeros((yhat.size, len(np.unique(yhat))))
            onehot[ np.arange(yhat.size), yhat ] = 1
            # onehot = np.zeros((yhat.size, np.max(yhat) + 1))
            # onehot[np.arange(yhat.size), yhat] = 1
            yhat = onehot
        
        return yhat

class NonLineaireSeq:
    """ Classe pour un classifieur non-linéaire par réseau de neurones,
        version séquentielle. La fonction fit correspond à la classe SGD
        demandée.
    """
    def fit(self, xtrain, ytrain, batch_size=1, neuron=10, niter=1000, gradient_step=1e-5):
        """ Classifieur non-linéaire sur les données d'apprentissage.
            @param xtrain: float array x array, données d'apprentissage
            @param ytrain: int array, labels sur les données d'apprentissage
            @param batch_size: int, taille des batchs
            @param neuron: nombre de neurones dans une couche
            @param niter: int, nombre d'itérations
            @param gradient_step: float, pas de gradient
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

        # Liste des couches du réseau de neurones
        modules = [ self.linear_1, self.tanh, self.linear_2, self.sigmoide ]

        # Apprentissage du réseau de neurones
        self.optim  = Optim(modules, self.mse, gradient_step)

        for i in range(niter):
            # Tirage d'un batch de taille batch_size et mise-à-jour
            inds = [ rd.randint(0, len(xtrain) - 1) for j in range(batch_size) ]
            self.optim.step( xtrain[inds], ytrain[inds] )
            self.list_loss.append(np.mean( self.mse.forward(ytrain, self.optim.predict(xtrain)) ))
        

    def SGD(self, net, loss, xtrain, ytrain, batch_size=1, niter=1000, gradient_step=1e-5):
        # Ajout d'un biais aux données
        xtrain = add_bias (xtrain)
        
        # Initialisation de la listte des loss
        self.list_loss = []
        
        # Apprentissage du réseau de neurones
        self.optim  = Optim(net, loss, gradient_step)
        
        # Liste de variables pour simplifier la création des batchs
        card = xtrain.shape[0]
        nb_batchs = card//batch_size
        inds = np.arange(card)

        # Création des batchs
        np.random.shuffle(inds)
        batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

        for i in range(niter):
            # On mélange de nouveau lorsqu'on a parcouru tous les batchs
            if i%nb_batchs == 0:
                np.random.shuffle(inds)
                batchs = [[j for j in inds[i*batch_size:(i+1)*batch_size]] for i in range(nb_batchs)]

            # Mise-à-jour sur un batch
            batch = batchs[i%(nb_batchs)]
            self.optim.step(xtrain[batch], ytrain[batch])
            current_loss = np.mean(loss.forward(ytrain, self.optim.predict(xtrain, onehot=True)))
            self.list_loss.append(current_loss)
         

    def predict(self, xtest):
        """ Prédiction sur des données. Il s'agit simplement d'un forward.
        """
        # Ajout d'un biais aux données
        xtest = add_bias (xtest)

        # Phase passe forward
        return self.optim.predict(xtest)


def test_non_lin_seq(neuron=10, niter=1000, gradient_step=1e-3, batch_size=None):    
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
    
    nonlin = NonLineaireSeq()
    nonlin.SGD(net, mse, datax, datay, batch_size=batch_size, niter=niter, gradient_step=gradient_step)
    
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