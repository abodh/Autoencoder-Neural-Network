# Neural Network Autoencoder

## Description
This is a simple example of using a neural network as an autoencoder without using any machine learning libraries in Python. 
The input is a 8-bit binary digits and as expected the output is the same 8-bit input values. There is a single hidden layer with 3 units/neurons. The figure below shows a similar autoencoder architecture [1]. However, we are only using a __single hidden neuron__ to maintain simplicity.


![Image](https://miro.medium.com/max/1968/1*44eDEuZBEsmG_TCAKRI3Kw@2x.png)


The feedforward and backpropagation algorithm equations, using the sigmoid function as an activation, have been derived and used in the code [2]. **The program is only designed for sigmoid as an activation function as al the derived equations are based on the assumptions of sigmoid as an activation function.** 


![Image](https://jamesmccaffrey.files.wordpress.com/2016/12/backpropgrad_05.jpg?w=547&h=&zoom=2)

Although popular, the training has not been carried on batches and hence, each of the training samples are trained separately. To assess the training process of the network the sum of squared error has been used as a metric. Along with that, the program also plots the learned weight parameters from the input layer to a single hidden unit which can be viewed on separate output folders. The user can modify the code to plot required weight parameters for different input and hidden neurons/layers.

## Defined parameters/hyperparameters:
* Number of input neurons = 8
* Number of hidden neurons = 3
* Number of hidden layers = 1
* Number of output neurons = 8
* Number of training iterations = 5000
* learning rate = 0.3 (a bit high but can be modified as per user's need)
* activation function = sigmoid

## Output

![Image](https://raw.githubusercontent.com/abodh/autoencoder-neural-network/master/output_plot/SSE.png)
![Image](https://raw.githubusercontent.com/abodh/autoencoder-neural-network/master/output_plot/Weights_i_h.png)
![Image](https://raw.githubusercontent.com/abodh/autoencoder-neural-network/master/output_plot/hidden_out.png)

## Dependencies
1. numpy
2. matplotlib
3. pandas

## References

[1] https://towardsdatascience.com/applied-deep-learning-part-3-autoencoders-1c083af4d798

[2] https://jamesmccaffrey.wordpress.com/2016/12/14/deriving-the-gradient-for-neural-network-back-propagation-with-cross-entropy-error/
