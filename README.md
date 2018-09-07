
# Activation Functions Lab

## Objective

In this lab, we'll learn about different common activation functions, and compare and constrast their effectiveness on an MLP for classification on the MNIST data set!

### Getting Started: What Is An Activation Function?

In your words, answer the following question:

**_What purpose do acvtivation functions serve in Deep Learning?  What happens if our neural network has no activation functions?  What role do activation functions play in our output layer? Which activation functions are most commonly used in an output layer?_**

Write your answer below this line:
______________________________________________________________________________________________________________________


For the first part of this lab, we'll only make use of the numpy library.  Run the cell below to import numpy.


```python
import numpy as np
```

## Writing Different Activation Functions

We'll begin this lab by writing different activation functions manually, so that we can get a feel for how they work.  

### Logistic Sigmoid Function


We'll begin with the **_Sigmoid_** activation function, as described by the following equation:

$$\LARGE \phi(z) = \frac{1}{1 + e^{-z}}  $$

In the cell below, complete the `sigmoid` function. This functio should take in a value and compute the results of the equation returned above.  


```python
def sigmoid(z):
    pass
```


```python
sigmoid(.458) # Expected Output 0.61253961344091512
```

### Hyperbolic Tangent (tanh) Function 

The hyperbolic tangent function is as follows:



$$\LARGE  \frac{e^x - e^{-x}}{e^x + e^{-x}}  $$

Complete the function below by implementing the `tanh` function.  


```python
def tanh(z):
    pass
```


```python
print(tanh(2)) # 0.964027580076
print(np.tanh(2)) # 0.0
print(tanh(0)) # 0.964027580076
```

### Rectified Linear Unit (ReLU) Function

The final activation function we'll implement manually is the **_Rectified Linear Unit_** function, also known as **_ReLU_**.  

The relu function is:

$$\LARGE  Max(0, z)  $$


```python
def relu(z):
    pass
```


```python
print(relu(-2)) # Expected Result: 0.0
print(relu(2)) # Expected Result: 2.0
```

### Softmax Function

The **_Softmax Function_** is primarily used as the activation function on the output layer for neural networks for multi-class categorical prediction.  The softmax equation is as follows:

<img src='softmax.png'>

The mathematical notation for the softmax activation function is a bit dense, and this is a special case, since the softmax function is really only used on the output layer. Thus, the code for the softmax function ahs been provided.  

Run the cell below to compute the softmax function on a sample vector.  


```python
z = [1.0, 2.0, 3.0, 4.0, 1.0, 2.0, 3.0]
softmax = np.exp(z)/np.sum(np.exp(z))
softmax
```

**_Expected Output:_**

array([ 0.02364054,  0.06426166,  0.1746813 ,  0.474833  ,  0.02364054,
        0.06426166,  0.1746813 ])


## Comparing Training Results 

Now that we have experience with the various activation functions, we'll gain some practical experience with each of them by trying them all as different hyperparameters in a neural network to see how they affect the performance of the model. Before we can do that, we'll need to preprocess our image data. 

We'll build 3 different versions of the same network, with the only difference between them being the activation function used in our hidden layers.  Start off by importing everything we'll need from Keras in the cell below.

**_HINT:_** Refer to previous labs that make use of Keras if you aren't sure what you need to import

### Preprocessing Our Image Data

We'll need to preprocess the MNIST image data so that it can be used in our model. 

In the cell below:

* Load the training and testing data and their corresponding labels from MNIST.  
* Reshape the data inside `X_train` and `X_test` into the appropriate shape (from a 28x28 matrix to a vector of length 784).  Also cast them to datatype `float32`.
* Normalize the data inside of `X_train` and `X_test`
* Convert the labels inside of `y_train` and `y_test` into one-hot vectors (Hint: see the [documentation](https://keras.io/utils/#to_categorical) if you can't remember how to do this).

### Model Architecture

Your task is to build a neural network to classify the MNIST dataset.  The model should have the following architecture:

* Input layer of `(784,)`
* Hidden Layer 1: 100 neurons
* Hidden Layer 2: 50 neurons
* Output Layer: 10 neurons, softmax activation function
* Loss: `categorical_crossentropy`
* Optimizer: `'SGD'`
* metrics:  `['accuracy']`

In the cell below, create a model that matches the specifications above and use a **_sigmoid activation function for all hidden layers_**.


```python
sigmoid_model = None

```

Now, compile the model with the following hyperparameters:

* `loss='categorical_crossentropy'`
* `optimizer='SGD'`
* `metrics=['accuracy']`

Now, fit the model.  In addition to our training data, pass in the following parameters:

* `epochs=10`
* `batch_size=32`
* `verbose=1`
* `validation_data=(X_test, y_test)`

## Fitting a Model with Tanh Activations

Now, we'll build the exact same model as we did above, but with hidden layers that use `tanh` activation functions rather than `sigmoid`.

In the cell below, create a second version of the model that uses hyperbolic tangent function for activations.  All other parameters, including number of hidden layers, size of hidden layers, and the output layer should remain the same. 


```python
tanh_model = None
```

Now, compile this model.  Use the same hyperparameters as we did for the sigmoid model. 

Now, fit the model.  Use the same hyperparameters as we did for the sigmoid model. 

## Fitting a Model with ReLU Activations

Finally, construct a third version of the same model, but this time with `relu` activation functions for the hidden layer.  


```python
relu_model = None
```

Now, compile the model with the same hyperparameters as the last two models. 

Now, fit the model with the same hyperparameters as the last two models. 

## Conclusion

Which activation function was most effective?


