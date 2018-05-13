import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import scipy.io as scio
from convnet import cnn
import signal
import time as t
import pickle


epoch_num = 5
train_num = 5
test_num = 1
batch_size = 1
test = True

#inicjalizacja sieci neuronowej
net = cnn()
#net.Weights = pickle.load(open('/Users/warbarbye/Desktop/PSZT/proj/weights.txt', 'rb'))
#net.Biases = pickle.load(open('/Users/warbarbye/Desktop/PSZT/proj/biases.txt', 'rb'))

#wczytanie obrazow oraz etykiet ze zbioru uczacego oraz testowego
mnist = scio.loadmat('/Users/warbarbye/Desktop/PSZT/proj/mnist_2D.mat')

numIter = 1
for epoch in range(epoch_num):
    #losowy wybor (train_num) oraz (test_num) numerow zdjec ze zbioru uczacego oraz zbioru testowego 
    input_train_ind = [np.random.randint(0,60000) for i in range(train_num)]
    input_test_ind = [np.random.randint(0,10000) for i in range(test_num)]
    #inicjalizacja zmiennych przechowujacych wczytane przyklady oraz etykiety ze zbioru uczacego
    label_train = np.asarray([[0 for i in range(10)] for j in range(train_num)]) 
    input_train = np.zeros((train_num, mnist['X_train'][0].shape[0], mnist['X_train'][0].shape[1]))
    #inicjalizacja zmiennych przechowujacych wczytane przyklady oraz etykiety ze zbioru testowego
    label_test = np.asarray([[0 for i in range(10)] for j in range(test_num)])
    input_test = np.zeros((test_num, mnist['X_train'][0].shape[0], mnist['X_train'][0].shape[1]))

    j=0
    #kodowanie (one hot) etykiet do wybranych zdjec
    #wczytanie zdjec ze zbioru
    for i in input_train_ind:
        label_train[j,mnist['Y_train'][i]] = 1
        input_train[j] = mnist['X_train'][i]
        j += 1
    j=0
    for i in input_test_ind:
        label_test[j,mnist['Y_test'][i]] = 1
        input_test[j] = mnist['X_test'][i]
        j += 1
    j = 0
    #wczytanie zdjec oraz etykiet do batcha oraz rozpoczecie procesu uczenia sie sieci
    while(j < train_num):
        batch_input = input_train[j:j+batch_size]
        batch_label = label_train[j:j+batch_size]
        batch_loss = net.backward(batch_input, batch_label)
        print('Iteration = {} \t  Train Loss = {} \n'.format(numIter ,batch_loss))
        numIter += 1
        j += batch_size
    acc = 0
    test_loss = 0
    #sprawdzenie rezultatow uczeia sieci po (train_num)-przykladach
    for i in range(test_num):
        imgplot = plt.imshow(input_test[i])
        plt.show()
        [predict,loss] = net.test_net(input_test[i], label_test[i])
        print("predict = {}".format(predict))
        if label_test[i][predict] == 1:
            acc += 1
        test_loss += loss
    if test:
        print('Epoch = {} \t Test Loss = {} \t Accuracy = {} '.format(epoch+1, test_loss/test_num, acc*100.0/test_num))
    if epoch == epoch_num-1:
        print("epoch {} and dumped!".format(epoch))
        pickle.dump(net.Weights, open('/Users/warbarbye/Desktop/PSZT/proj/weights.txt', 'wb'))
        pickle.dump(net.Biases, open('/Users/warbarbye/Desktop/PSZT/proj/biases.txt', 'wb'))
