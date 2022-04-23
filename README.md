# Linear Regression train by Gradient Descent and Visualize It
## Description
* Use Python Numpy and matplotlib, simply implement Linear Regression and show training weight trace in each step <br>also visualize with contour figure.
## Implement Details
* We need to use a function <svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" width="39.152396pt" height="9.009766pt" viewBox="0 0 39.152396 9.009766" version="1.1">
<defs>
<g>
<symbol overflow="visible" id="glyph0-0">
<path style="stroke:none;" d=""/>
</symbol>
<symbol overflow="visible" id="glyph0-1">
<path style="stroke:none;" d="M 7.59375 -7.84375 L 5.328125 -7.84375 L 5.328125 -7.640625 L 5.671875 -7.609375 C 6.03125 -7.578125 6.15625 -7.46875 6.15625 -7.265625 C 6.15625 -6.953125 5.625 -6.203125 4.375 -4.71875 L 3.90625 -4.15625 L 3.53125 -5.484375 C 3.234375 -6.5 3.09375 -7.03125 3.09375 -7.21875 C 3.09375 -7.46875 3.234375 -7.578125 3.953125 -7.640625 L 3.953125 -7.84375 L 1.09375 -7.84375 L 1.09375 -7.640625 C 1.765625 -7.546875 1.84375 -7.453125 2.015625 -6.828125 L 2.921875 -3.625 L 2.21875 -1.15625 C 2 -0.390625 1.734375 -0.234375 0.9375 -0.1875 L 0.9375 0 L 4.40625 0 L 4.40625 -0.1875 L 3.96875 -0.234375 C 3.5 -0.265625 3.34375 -0.375 3.34375 -0.671875 C 3.34375 -0.921875 3.421875 -1.25 3.703125 -2.203125 L 4.125 -3.671875 L 6.75 -6.90625 C 7.15625 -7.40625 7.25 -7.53125 7.59375 -7.640625 Z M 7.59375 -7.84375 "/>
</symbol>
<symbol overflow="visible" id="glyph0-2">
<path style="stroke:none;" d="M 10.875 -7.84375 L 8.65625 -7.84375 L 8.65625 -7.640625 C 9.265625 -7.625 9.46875 -7.4375 9.46875 -7.140625 C 9.46875 -6.90625 9.34375 -6.625 9.1875 -6.328125 L 6.9375 -1.9375 L 6.453125 -6.890625 C 6.453125 -6.921875 6.4375 -7 6.4375 -7.0625 C 6.4375 -7.453125 6.640625 -7.578125 7.25 -7.640625 L 7.25 -7.84375 L 4.421875 -7.84375 L 4.421875 -7.640625 C 5.109375 -7.625 5.25 -7.515625 5.3125 -6.90625 L 5.40625 -6.125 L 3.34375 -1.9375 L 2.8125 -6.796875 C 2.8125 -6.875 2.8125 -6.96875 2.8125 -7.03125 C 2.8125 -7.484375 3.0625 -7.578125 3.671875 -7.640625 L 3.671875 -7.84375 L 0.859375 -7.84375 L 0.859375 -7.640625 C 1.234375 -7.59375 1.359375 -7.578125 1.46875 -7.46875 C 1.625 -7.28125 1.6875 -7.078125 1.8125 -5.953125 L 2.546875 0.21875 L 2.765625 0.21875 L 5.4375 -5.21875 L 5.5 -5.21875 L 6.09375 0.21875 L 6.34375 0.21875 L 9.953125 -6.75 C 10.296875 -7.40625 10.421875 -7.5 10.875 -7.640625 Z M 10.875 -7.84375 "/>
</symbol>
<symbol overflow="visible" id="glyph1-0">
<path style="stroke:none;" d=""/>
</symbol>
<symbol overflow="visible" id="glyph1-1">
<path style="stroke:none;" d="M 7.640625 -3.84375 L 7.640625 -4.625 L 0.578125 -4.625 L 0.578125 -3.84375 Z M 7.640625 -1.4375 L 7.640625 -2.234375 L 0.578125 -2.234375 L 0.578125 -1.4375 Z M 7.640625 -1.4375 "/>
</symbol>
<symbol overflow="visible" id="glyph2-0">
<path style="stroke:none;" d=""/>
</symbol>
<symbol overflow="visible" id="glyph2-1">
<path style="stroke:none;" d="M 4.03125 -1.171875 L 3.921875 -1.21875 C 3.640625 -0.734375 3.453125 -0.640625 3.09375 -0.640625 L 1.109375 -0.640625 L 2.515625 -2.140625 C 3.265625 -2.953125 3.609375 -3.578125 3.609375 -4.25 C 3.609375 -5.09375 2.984375 -5.75 2.03125 -5.75 C 0.984375 -5.75 0.4375 -5.0625 0.25 -4.0625 L 0.4375 -4.015625 C 0.78125 -4.859375 1.078125 -5.125 1.6875 -5.125 C 2.40625 -5.125 2.875 -4.703125 2.875 -3.921875 C 2.875 -3.203125 2.5625 -2.546875 1.765625 -1.71875 L 0.25 -0.109375 L 0.25 0 L 3.578125 0 Z M 4.03125 -1.171875 "/>
</symbol>
</g>
</defs>
<g id="surface1">
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-1" x="-0.269531" y="8.792969"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph1-1" x="11.414062" y="8.796875"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph0-2" x="22.878906" y="8.792969"/>
</g>
<g style="fill:rgb(0%,0%,0%);fill-opacity:1;">
  <use xlink:href="#glyph2-1" x="34.640625" y="5.757812"/>
</g>
</g>
</svg> to fit Features(x_data) and Target(y_data)
* Here we use gradient descent to update ours weights
* Loss Function: $MSE=\sum\limits_{i=1}^{n}(y_i-\hat{y}_i)^2$ 
* Gradient: Calculate partial differential of Loss Fucntion
    * Let $ L = \sum\limits_{i=1}^{n}(y_i-\hat{y}_i)^2 = \sum\limits_{i=1}^{n}(y_i-(wx+b))^2 $
    * $ \frac{\partial L}{\partial w} = \sum\limits_{i=1}^{n}2x_i(b_i+w_ix_i-y_i) $
    * $\frac{\partial L}{\partial b} = \sum\limits_{i=1}^{n}2(b_i+w_ix_i-y_i)$


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


