import numpy as np
from scipy import signal
import activations as act
from forward_layer import *
from back_propagation import *


inp_width = 28
inp_height = 28
inp_channels = 1
layers = [('conv',6, 5, 1), ('pool',2,2), ('conv',16,5, 1), ('pool',2,2), ('fc',120), ('fc',84), ('softmax',10)]
initBias = 0.01
eta = 0.5
alpha = 0.9

class cnn:
    """Klasa konwolucyjnej sieci neuronowej"""
    def __init__(self):
        """Konstruktor konwolucyjnej sieci neuronowej. W kazdym wierszu inizjalizowana jest kolejna warstwa sieci """
        #conv1
        n = inp_width*inp_height
        #poczatkowe wagi sieci sa ustalane losowo z rozkladu normalnego. Umieszczane sa one na liscie matryc wag
        self.Weights = [np.random.randn(layers[0][1],inp_channels,layers[0][2],layers[0][2])/np.sqrt(n)]
        out_Size =  inp_width - layers[0][2] + 1 #zmienna zawiera rozmiar wyjscia danej warstwy
        #inicjalizacja progow 
        self.Biases = [initBias*np.ones( layers[0][1] )]
        #przypisanie parametrow warstwie poolingu
        self.poolParams = [(layers[1][1], layers[1][2])]
        out_Size = out_Size/2 
        #conv 2
        n = out_Size*out_Size*layers[0][1]
        self.Weights.append(np.random.randn(layers[2][1],layers[0][1],layers[2][2],layers[2][2])/np.sqrt(n))
        out_Size = out_Size - layers[2][2]+1
        self.Biases.append(initBias*np.ones(layers[2][1]))
        #pool 2
        self.poolParams.append((layers[3][1],layers[3][2]))
        out_Size = out_Size/2 
        #conv 3
        n = out_Size*out_Size*layers[2][1]
        self.Weights.append(np.random.randn(layers[4][1],layers[2][1],out_Size,out_Size)/np.sqrt(n))
        out_Size = 1
        self.Biases.append(initBias*np.ones(layers[4][1]))
        #fully connected 1
        n = layers[4][1]
        self.Weights.append(np.random.randn(layers[5][1],layers[4][1])/np.sqrt(n))
        self.Biases.append(initBias*np.ones(layers[5][1]))
        #fully connected 2
        n = layers[5][1]
        self.Weights.append(np.random.randn(layers[6][1],layers[5][1])/np.sqrt(n))
        self.Biases.append(initBias*np.ones(layers[6][1]))

        self.Weights = np.asarray(self.Weights)
        self.Biases = np.asarray(self.Biases)
        
        delta_W = []
        delta_B = []
        for i in range(5):
            delta_W.append(np.zeros(self.Weights[i].shape))
            delta_B.append(np.zeros(self.Biases[i].shape))
        self.delta_W = np.asarray(delta_W)
        self.delta_B = np.asarray(delta_B)

    def forward(self, inputData):
        """Funkcja realizacje propagacje danych w przod
        Wejscie: obraz lub paczka obrazow ze zbioru danych
        Wyjscie: produkt warstwy softmax: 10-dim wektor liczb
        """
        weights = self.Weights
        biases = self.Biases
        poolParams = self.poolParams
        cache = [] #zmienna przechowujaca produkty warstw - pomocna do propagacji wstecznej
        #warstwa wejsciowa
        layer0 = np.asarray(inputData)
        cache.append(layer0)
        #pierwsza warstwa konwolucyjna
        layer1 = convolution_forward(np.asarray([layer0]),weights[0],biases[0])
        cache.append(layer1)
        #pierwsza warstwa max poolingu
        layer2 = maxpool_forward(layer1, poolParams[0][0], poolParams[0][1])
        cache.append(layer2)
        #druga warstwa konwolucyjna
        layer3 = convolution_forward(layer2,weights[1],biases[1])
        cache.append(layer3)
        #druga warstwa max poolingu
        layer4 = maxpool_forward(layer3, poolParams[1][0], poolParams[1][1])
        cache.append(layer4)
        #pierwsza warstwa fully connected zrealizowana jako warstwa konwolucyjna
        layer5 = convolution_forward( layer4,weights[2] ,biases[2] )
        cache.append(layer5)
        #druga warstwa fully connected z funkcja aktywacji typu ReLU
        layer6 = act.relu(np.dot(weights[3],layer5[:,0]).transpose() + biases[3]).transpose()
        cache.append(layer6)
        #softmax
        layer7 = np.dot( weights[4], layer6[:,0] ).transpose() + biases[4]
        layer7 -= np.max(layer7)
        layer7 = np.exp(layer7)/sum(np.exp(layer7))

        return (layer7, cache)

    def test_net(self, input_test, input_test_label):
        """Funkcja sprawdzajaca efekty uczenia sie sieci neuronowej
        Wejscie: obrazy ze zbioru testowego, etykiety ze zbioru testowego
        Wyjscie: Wektor kodujacy rozpoznana (lub nie) cyfre z obrazu, wartosc funkcji straty
        """
        out = self.forward(input_test)[0]
        loss = -1*sum(input_test_label * np.log(out))

        return [out.argmax(),loss]

    def backward(self, input_train, input_train_label):
        """Funkcja realizujaca caly proces uczenia sie sieci neuronowej. Propagacja w przod, propagacja wsteczna oraz aktualizacja parametrow warstw.
        Wejscie: obrazy ze zbioru uczacego, etykiety ze zbioru uczacego
        """
        batchSize = len(input_train) #liczba obrazow podawanych na wejscie w trakcie jednej iteracji
        weights = self.Weights
        biases = self.Biases
        delta_W = self.delta_W
        delta_B = self.delta_B
        poolParams = self.poolParams
        dW_list = []
        dB_list = []
        dW4 = np.zeros(weights[4].shape)
        dB4 = np.zeros(biases[4].shape)
        dW3 = np.zeros(weights[3].shape)
        dB3 = np.zeros(biases[3].shape)
        dW2 = np.zeros(weights[2].shape)
        dB2 = np.zeros(biases[2].shape)
        dW1 = np.zeros(weights[1].shape)
        dB1 = np.zeros(biases[1].shape)
        dW0 = np.zeros(weights[0].shape)
        dB0 = np.zeros(biases[0].shape)
        loss = 0
        for image in range(batchSize):

            X_data = input_train[image]
            X_label = input_train_label[image]
            output_forward, cache = self.forward(X_data)         
            loss += -1*sum(X_label - np.log(output_forward)) #obliczenie wartosci funkcji straty [cross entropy]

            #Propagacja wsteczna gradientu
            dy = -1*(X_label - output_forward)/2
            #print("X_label = {} \t layer7 = {} \t dy = {}".format(X_label, output_forward, dy))

            [dy, dW, dB ] = fullycon_b(cache[6], np.asarray([dy]).transpose() , weights[4])
            dW4 += dW
            dB4 += dB.flatten() #wektoryzacja macierzy
            dy = act.relu_b(dy.transpose(), cache[6])

            [dy, dW, dB ] = fullycon_b(cache[5][:,0], dy, weights[3])
            dW3 += dW
            dB3 += dB.flatten()
            dy = act.relu_b(dy.transpose(), cache[5][:,0]) 
            
            [dy, dW, dB ] = convolution_b(cache[4], dy, weights[2])
            dW2 += dW
            dB2 += dB.flatten()
            
            dy = maxpool_b(cache[3], dy)
            dy = act.relu_b(dy, cache[3])

            [dy, dW, dB ] = convolution_b(cache[2], dy, weights[1])
            dW1 += dW
            dB1 += dB.flatten()
            
            dy = maxpool_b(cache[1], dy)
            dy = act.relu_b(dy, cache[1]) 

            [dy, dW, dB ] = convolution_b(np.asarray([cache[0]]), dy, weights[0])
            dW0 += dW
            dB0 += dB.flatten()
			
        dW_list.append(dW4)
        dB_list.append(dB4)
        dW_list.append(dW3)
        dB_list.append(dB3)
        dW_list.append(dW2)
        dB_list.append(dB2)
        dW_list.append(dW1)
        dB_list.append(dB1)
        dW_list.append(dW0)
        dB_list.append(dB0)
        dW_list = dW_list[::-1]
        dB_list = dB_list[::-1]
            
        #Aktualizacja parametrow kazdej z warstw (o ile takie posiada)
        #uczenie z metoda momentum: learning rate = const; alpha = const
        for x in range(len(dW_list)):
            delta_W[x] = alpha*delta_W[x] - eta*dW_list[x]/batchSize
            weights[x] += delta_W[x]
            delta_B[x] = alpha*delta_B[x] - eta*dB_list[x]/batchSize
            biases[x] += delta_B[x]
        #przypisanie nowych wag po aktualiacji wszystkich parametrow
        self.Weights = weights
        self.Biases = biases

        #zwrocenie stosunku wartosci f-cji straty do rozmiaru batch'u
        return loss/batchSize




