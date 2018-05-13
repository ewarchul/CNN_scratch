from scipy import signal
import numpy as np
import activations as act

def convolution(X, filters):
    """Funkcja wykonujaca operacje splotu.
	Wejscie: obiekt lub grupa obiektow, lista filtrow
	Wyjscie: tablica przechowujaca listy wszystkich produktow splotu X z filtrami
    """
    feature_maps = [] #lista list
    for i in range(len(filters)):
        feature_map = [] #lista, ktora przechowuje wynik splotu dla danego filtru
        filter1 = filters[i]
        depth = len(filter1)
        for j in range(depth):
            feature_map.append(signal.convolve2d(X[j], np.rot90(filter1[j],2),'valid')) #do wykonania operacji splotu uzyto funkcji z pakietu SciPy: rot90(matrix, 2) == rot180

        feature_maps.append(sum(feature_map))

    return np.asarray(feature_maps)


def convolution_b(X, dy, W):

    """Funkcja obliczajaca propagacje wsteczna dla wartwy konwolucyjnej.
	Wejscie: obiekt lub grupa obiektow (feature map), gradient z poprzedniej warstwy, wagi
	Wyjscie: gradient, pochodna wag, pochodna progow
	"""
	#reshape wektora gradientu w macierz 3d 
    if dy.shape[1] == 1:
        temp = dy
        dy = np.zeros((dy.shape[0],1,1))
        dy[:,0] = temp
	 
    Wb = np.zeros((W.shape[1],W.shape[0],W.shape[2],W.shape[3]))
    dW = []
    dB = []
	
    for i in range(W.shape[0]):
        kernel = []
        for j in range(W.shape[1]):
            kernel.append(signal.convolve2d(X[j], np.rot90(dy[i],2) ,'valid'))
            Wb[j,i] = np.rot90(W[i,j],2)
        dW.append(np.asarray(kernel))
        dB.append(np.sum(dy[i]))

    pad = W.shape[2]-1
    dy = np.pad(dy, pad ,'constant' )[pad:-1*pad]

    dX = convolution(dy, Wb) #obliczenie gradientu

    dW = np.asarray(dW)
    dB = np.asarray(dB)

    return [dX, dW, dB]

def fullycon_b(X, dy, W):
    """Funkcja realizujaca propagacje wsteczna dla warstwy fully connected
	Wejscia:  obiekt lub grupa obiektow (feature map), gradient z poprzedniej warstwy, wagi
	Wyjscia: gradient, pochodna wag i pochodna progow 
	"""
    dX = np.dot(dy.transpose(), W)
    dW = np.dot(dy, X.transpose())
    dB = dy

    return [ dX, dW, dB ]

def maxpool_b(X, dy):
    """Funkcja realizujaca propagacje wsteczna dla wartwy poolingu
    Wejscia: obiekt lub grupa obiektow, gradient z poprzedniej warstwy
	Wyjscia: gradient
    """
    dX = np.zeros(X.shape)
	#stride = 2
	#kernel = 2
    for k in range(X.shape[0]):
        for i in range(0,X.shape[1],2):
            for j in range(0,X.shape[2],2):
                a =  X[k,i:i+2,j:j+2]
                ind = np.unravel_index(a.argmax(), a.shape) #indeksy wartosci maksymalnych dla odpowiedniego fragmentu obiektu wejsciowego
                dX[k,i+ind[0],j+ind[1]] = dy[k,i/2,j/2]

    return dX


