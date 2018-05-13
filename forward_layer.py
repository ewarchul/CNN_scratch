from scipy import signal
import numpy as np
import activations as act


def convolution_forward(X, filters, bias):
    """Funckja realizujaca warstwe konwolucyjna z liniowa funkcja aktywacji typu ReLU
    Jest to ta sama f-cja co convolution w pliku splot.py z ta roznica, ze bierze pod uwage macierz progow
    """	
    feature_maps = []
    for i in range(len(filters)):
        feature_map = []
        filter1 = filters[i]
        depth = len(filter1)
        for j in range(depth):
            feature_map.append(signal.convolve2d(X[j], np.rot90(filter1[j],2) ,'valid'))
        feature_map = sum(feature_map) + bias[i]*np.ones((feature_map[0].shape[0], feature_map[0].shape[1]))
        feature_maps.append(act.relu(feature_map))
    return np.asarray(feature_maps)

def maxpool_forward(X, kernel=2, stride=2):
    """Funkcja realizujaca warstwe max-poolingu.
    Wejscie: obiekt lub grupa obiektow, parametry poolingu: stride oraz rozmiar.
    Wyjscie: tablica tensorow po wykonaniu poolingu o odpowiednio mniejszym rozmiarze""" 
    pool_ret = []
    for i in range(len(X)):
        temp = X[i].reshape(X[i].shape[0]/2,2,X[i].shape[1]/2,2) #zmiana rozmiaru 
        pool_ret.append(temp.max(axis=(1,3))) #kluczowa operacja max poolingu - wyekstrahowanie wartosci maskymalnej
    return np.asarray(pool_ret)


#xx = np.random.rand(6, 6)
#xx = np.array([xx])
#print(xx)
#print(xx.shape)
#y = maxpool_forward(xx, 2, 2)
#print(y)

