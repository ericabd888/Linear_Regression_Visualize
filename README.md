# Linear Regression train by Gradient Descent and Visualize It
## Description
* Use Python Numpy and matplotlib, simply implement Linear Regression and show training weight trace in each step <br>also visualize with contour figure.
## Implement Details
* We need to use a function <img src="https://latex.codecogs.com/svg.image?\color{Gray}{&space;Y&space;=&space;WX&space;&plus;&space;B&space;}"> to fit Features(x_data) and Target(y_data)
* Here we use gradient descent to update ours weights
* Loss Function: <img src="https://latex.codecogs.com/svg.image?\color{Gray}{&space;MSE=\sum\limits_{i=1}^{n}(y_i-\hat{y}_i)^2&space;}">
* Gradient: Calculate partial differential of Loss Fucntion
    * Let <img src="https://latex.codecogs.com/svg.image?\color{Gray}{&space;L&space;=&space;\sum\limits_{i=1}^{n}(y_i-\hat{y}_i)^2&space;=&space;\sum\limits_{i=1}^{n}(y_i-(wx&plus;b))^2&space;}">
    * <img src="https://latex.codecogs.com/svg.image?\color{Gray}{\frac{\partial&space;L}{\partial&space;w}&space;=&space;\sum\limits_{i=1}^{n}2x_i(b_i&plus;w_ix_i-y_i)}">
    * <img src="https://latex.codecogs.com/svg.image?\color{Gray}{\frac{\partial&space;L}{\partial&space;b}&space;=&space;\sum\limits_{i=1}^{n}2(b_i&plus;w_ix_i-y_i)}">


## Code OutLook:
```python
# initial weight and bias
b=-120
w=-4
# base learning rate
lr=1.3
# iteration
iteration=50000
# store each weight and bias to plot contour figure
b_history=[b]
w_history=[w]
# Here is the trick for AdaGrad(Adavance Learning Rate Trick)
lr_b=0.0
lr_w=0.0
for i in range(iteration):
    # zero gradient first
    b_grad=0.0  
    w_grad=0.0   
    # compute all data
    for n in range(len(x_data)):
        # the partial differential of L(w,b) with respect to b
        b_grad = b_grad -2.0*(y_data[n] - b - w*x_data[n] )*1.0
        # the partial differential of L(w,b) with respect to weight
        w_grad = w_grad -2.0*(y_data[n] - b - w*x_data[n] )*x_data[n] 
    # Bellow is the trick for AdaGrad(Adavance Learning Rate Trick)
    lr_b = lr_b + b_grad **2
    lr_w = lr_w + w_grad **2
    b = b - lr/np.sqrt(lr_b)*b_grad # use Adagrad
    w = w - lr/np.sqrt(lr_w)*w_grad
    # Above is the trick for AdaGrad(Adavance Learning Rate Trick)
    # Below: store each weight and bias
    b_history.append(b)
    w_history.append(w)
```

## Result
* We can see the training step is getting thicker, and red fork is target(which Loss is the lowest one)<br>
<img src="https://i.imgur.com/b0YnARZ.png" width=280 align="left"/>


