#Logistic Regression containing 2 datasets

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import make_moons


X, y = make_moons(n_samples = 1000, noise = 0.24)


#sigmoid function
def sigmoid(z):
    return 1.0/(1 + np.exp(-z))


#to find cost function
def costfunc(y , h):
    J = -np.mean(y*(np.log(h)) - (1-y)*np.log(1-h))
    return J

#to find gradient
def gradient(X , y , h):
    m = X.shape[0]
    dw = (1/m)*np.dot(X.T, (h - y))  #X.T calculates transpose , np.dot does matrix multiply
    db = (1/m)*np.sum((h - y))
    return dw , db

#plot
def plot_boundary(X , theta , bias):
    x1 = [min(X[:,0]), max(X[:,0])]
    m = -theta[0]/theta[1]
    c = -bias/theta[1]
    x2 = m*x1 + c

    plt.figure(figsize=(10,8))
    plt.plot(X[:, 0][y == 0], X[:, 1][y == 0], 'ro')
    plt.plot(X[:, 0][y == 1], X[:, 1][y == 1], 'bo')
    plt.xlim([-2 , 2])
    plt.ylim([-1 , 2])
    plt.xlabel("feature 1")
    plt.ylabel("feature 2")
    plt.title('Decision Boundary')

    plt.plot(x1, x2, 'y-')
    #plt.show()


#normalize
def normalize(X):
    m , n = X.shape
    
    # Normalizing all the n features of X.
    for i in range(n):
        X = (X - X.mean(axis=0))/X.std(axis=0)
        
    return X

#to train the dataset
def train(X , y , size , num_iter , alpha):
    
    m, n = X.shape

    # Initializing weights and bias to zeros.
    theta = np.zeros((n,1))
    bias = 0
    
    # Reshaping y.
    y = y.reshape(m,1)
    
    # Normalizing the inputs.
    x = normalize(X)
    
    # Empty list to store losses.
    arr = []

    for itr in range(num_iter):
        for i in range((m-1)//size + 1):
            
            start_i = i*size
            end_i = start_i + size
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            
            # Calculating hypothesis/prediction.
            h = sigmoid(np.dot(xb , theta) + bias)
            
            # Getting the gradients of loss w.r.t parameters.
            dtheta, dbias = gradient(xb, yb, h)
            
            # Updating the parameters.
            theta -= alpha * dtheta
            bias -= alpha * dbias
        
        # Calculating costfunction and appending it in the list.
        cost = costfunc(y , sigmoid(np.dot(xb , theta) + bias))
        arr.append(cost)
        
    # returning weights, bias and losses(List).
    return theta , bias , arr

theta , bias , l = train(X , y , size = 1000, num_iter = 1000 , alpha = 0.01)

# Plotting Decision Boundary
plot_boundary(X , theta , bias)

plt.show()


