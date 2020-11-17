Data Preprocessing
------------------


The fields, Delivery Phase, Education and Residence had 4, 3 and 2 missing values respectively.
Further analysing them, the values are mostly repeated, thus we went with filling those missing values with the Mode of the particular column.

Assuming Haemoglobin content had some correlation with the Community, Age, and their Blood Pressure, we grouped their values by these fields respectively, and filled the missing values with the group mean.
And to handle misisng values in the particular columns, decreased the features by one, to handle any missing data. For example, if blood pressure was also missing along with Haemoglobin, we took the group mean using Community and Age only.

A similar approach was taken for the Blood Pressure column as well.

For Age, from analysing the data, we chose to take the mean of the community mean to fill in missing age values.
And the similar approach for the weight column as well.

In the end, as the result set values were skewed towards 1 (75-25 split), we performed oversampling by duplicating the 0 values, to match the data spread.


Neural Network Architecture
---------------------------


We used a 3-layer neural network with Dense layers of size [9, 20], [20, 20] and [20, 1] (in the form (input, output) shape).

The first two layers used ReLU (Rectified Linear Unit) and LeakyReLU (Leaky Rectified Linear Unit) activation functions respectively whereas the final layer used Sigmoid activation function.
We also implemented Xavier initialisation (divide the weights by sqrt(2 / in_len)).

We used Binary Cross Entropy as the loss function.

We used Batch Gradient Descent optimizer with L2 regularization.


Hyperparameters
---------------


Learning rate: 0.05
Lambda parameter of L2 regularization: 0.01
Number of training steps: 1000

Activation functions: [ReLU, LeakyReLU, Sigmoid]
LeakyReLU parameter: 0.01
Initialisation: [Xavier initialisation, Xavier initialisation, random normal]
Length of vectors at each layer: [9, 20, 20, 1]


Implementation Details
----------------------

We have tried to mimic the PyTorch API of Sequential and Dense layers, Losses as well as optimizers.

We believe this is a key feature of our implementation as it makes the code very modular and extensible. Along with this, we also implemented L2 regularization, LeakyReLU and momentum which are some more extra features we implemented.

Neural network:
 - Used np.einsum (Einstein summation) for matric manipulation
 - Param class that tracks values as well as gradients

Losses:
 - Used a small epsilon value for numeric stability in divide and log

Optimizer:
 - Gradient descent with batches
 - L2 regularization
 - Momentum (beta1 parameter) - this was not used in the final model


Final Results
-------------

Train accuracy: 90.741%
Test accuracy: 97.222%

Confusion Matrix :
[[22, 1 ],
 [0,  13]]


Precision : 0.9285714285714286
Recall : 1.0
F1 SCORE : 0.962962962962963


Steps to run
------------

cd src/
python Neural_Net.py

