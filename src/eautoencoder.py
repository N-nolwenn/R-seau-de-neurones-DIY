from dmulti import *

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

class BCE (Loss) :
    """ Classe pour la fonction de coût cross-entropique binaire.
    """
    def forward (self, y, yhat):
        return -( y * np.maximum( -100, np.log( yhat + 0.01 ) ) + ( 1 - y ) * np.maximum( -100, np.log( 1 - yhat + 0.01 ) ) )

    def backward (self, y, yhat) :
        return - ( ( y / ( yhat + 0.01 ) )- ( ( 1 - y ) / ( 1 - yhat + 0.01 ) ) )

class AutoEncodeur :
    """ Classe pour l'auto-encodage (réduction des dimensions, compression de 
        l'information).
    """
    def codage (self, xtrain, modules):
        """ Phase d'encodage.
        """
        res_forward = [ modules[0].forward(xtrain) ]

        for j in range(1, len(modules)):
            #print("AFFICHAGE",type(modules[j]).__name__,res_forward[-1])

            res_forward.append( modules[j].forward( res_forward[-1] ) )

        return res_forward

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
        #xtrain = add_bias (xtrain)

        # Récupération des tailles des entrées
        batch, output = ytrain.shape
        batch, input = xtrain.shape

        # Initialisation des couches du réseau et de la loss
        self.bce = BCE()
        self.linear_1 = Linear(input, neuron)
        self.tanh = TanH()
        self.linear_2 = Linear(neuron, output)
        self.sigmoide = Sigmoide()
        self.linear_3 = Linear (output, neuron)
        self.linear_4 = Linear (neuron, input)


        # Liste des couches du réseau de neurones
        self.modules_enco = [ self.linear_1, self.tanh, self.linear_2, self.tanh ]
        self.modules_deco = [ self.linear_3, self.tanh, self.linear_4, self.sigmoide ]
        self.net = self.modules_enco + self.modules_deco

        for i in range(niter):
            res_forward_enco = self.codage(xtrain, self.modules_enco)
            res_forward_deco = self.codage(res_forward_enco[-1], self.modules_deco)
            res_forward = res_forward_enco + res_forward_deco
            #print("SIG", res_forward[-1])
            #print("RES FORWARD:",res_forward)
            if(i%100==0):
                print(np.sum(np.mean(self.bce.forward(xtrain, res_forward[-1]), axis=1)))

            # Phase backward (rétro-propagation du gradient de la loss
            #          par rapport aux paramètres et aux entrées)

            deltas =  [ self.bce.backward( xtrain, res_forward[-1] ) ]

            for j in range(len(self.net) - 1, 0, -1):
                deltas += [self.net[j].backward_delta( res_forward[j-1], deltas[-1] ) ]


            #Phase backward par rapport aux paramètres et mise-à-jour
            for j in range(len(self.net)):
                # Mise-à-jour du gradient
                if j == 0:
                    self.net[j].backward_update_gradient(xtrain, deltas[-1])
                else:
                    self.net[j].backward_update_gradient(res_forward[j-1], deltas[-j-1])

                # Mise-à-jour des paramètres
                self.net[j].update_parameters(gradient_step)
                self.net[j].zero_grad()

    def predict (self, xtest) :
        """ Prédiction sur des données de test.
        """
        res_forward_enco = self.codage(xtest, self.modules_enco)
        res_forward_deco = self.codage(res_forward_enco[-1], self.modules_deco)
        return res_forward_enco[-1], res_forward_deco[-1]

def test_auto_encodeur_():
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
    
    ## One-Hot Encoding
    onehot = np.zeros((ytrain.size,classes))
    onehot[ np.arange(ytrain.size), ytrain ] = 1
    ytrain = onehot
    
    # Initialisation de l'auto-encodeur
    auto = AutoEncodeur()
    auto.fit(xtrain, ytrain, niter=500, neuron=100, gradient_step=1e-4)
    #ytrain = np.argmax(ytrain, axis=1)
    
    # Test sur les données d'apprentissage
    y_enco,y_deco= auto.predict(xtest)

def usps_data(classes=10, a_bruit=False):
    # Chargement des données USPS
    uspsdatatrain = "data/USPS_train.txt"
    uspsdatatest = "data/USPS_test.txt"
    
    alltrainx,alltrainy = load_usps(uspsdatatrain)
    alltestx,alltesty = load_usps(uspsdatatest)
    
    # Création des données d'apprentissage et des données d'entraînement
    xtrain, ytrain = get_usps([i for i in range(classes)], alltrainx, alltrainy)
    xtest, ytest = get_usps([i for i in range(classes)], alltestx, alltesty)
    
    if a_bruit :
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

def test_auto_encodeur(neuron = 100, classes=10, niter=500, gradient_step=1e-4):
    """
        Tests de la partie auto-encodeur sur les données manuscrites (usps).
    """
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
    
    ## One-Hot Encoding
    onehot = np.zeros((ytrain.size,classes))
    onehot[ np.arange(ytrain.size), ytrain ] = 1
    ytrain = onehot
    
    time_start = time.time()
    
    # pour auto-encodeur
    auto = AutoEncodeur()
    auto.fit(xtrain, ytrain, ytrain.shape[1], niter=niter, neuron=neuron, gradient_step=gradient_step)
    
    # Test sur les données d'apprentissage
    y_enco,y_deco= auto.predict(xtest)
    
    kmeans = KMeans(n_clusters = 10, init='k-means++', n_init=10, max_iter=300, tol=0.0001, random_state=0)
    pred_enco = kmeans.fit_predict(y_enco)
    pred_deco = kmeans.fit_predict(y_deco)
    
    liste_test = [205,540,780,863,1027,1312,1505,1576,1741,1835]
    
    plt.figure(figsize=(15,4))

    for i in range(10):
        x = liste_test[i]
        print(x)
        plt.subplot(2,10, i+1)
        show_usps(xtest[x])
        plt.subplot(2,10, i+11)
        show_usps(y_deco[x])
        
    plt.savefig("auto_encodeur_bce.pdf")
    plt.show()  
    
    #t-sne
    perplexity_list = [5, 10, 30, 50, 100]
    tsne_perp = [ TSNE(n_components=2, random_state=0, perplexity=perp) for perp in perplexity_list]
    tsne_perp_data = [ tsne.fit_transform(y_deco) for tsne in tsne_perp]
    plt.figure(figsize=(15,15))
    for i in range(len(perplexity_list)):
        plt.subplot(3,2, i+1)
        plt.title("Perplexity " + str(perplexity_list[i]))
        plt.scatter(tsne_perp_data[i][:,0],tsne_perp_data[i][:,1], c=ytest, label=ytest)
    plt.subplot(3,2, i+2)
    plt.axis('off')
    plt.legend()
    print("\nTemps d'exécution: ", time.time() - time_start )