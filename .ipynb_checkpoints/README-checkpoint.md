# BikeShare Riders Prediction

![](images/markus-winkler-unsplash-bikes.jpg)

### Predict Number of BikeShare riders on a given day


## Table of Contents

- [Introduction](#introduction) 
- [Objective](#objective)
- [Dataset](#dataset)
- [Solution Approach](#solution-approach)
- [Evaluation Criteria](#evaluation-criteria)
- [How To Use](#how-to-use)
- [License](#license)
- [Author Info](#author-info)

---

## Introduction
Imagine a bike sharing company called BikeShare that rents bikes to riders. Company's revenue/profit is directly related with how many bikes it rents out on a given day. A major dilemma that company faces is to forecast how many bikes it needs to make available in the shop on a given day. BikeShare will loose potential business and revenue if there are too many riders asking for bikes but there aren't enough bikes available for rent. On the other hand if riders are too few then extra bikes will be just sitting in the shop without being used and loosing money. If BikeShare somehow, looking at the historical rental data, could forecast the number of riders it can expect looking to rent the bike on a given day then it can use this prediction to efficiently plan and manage the number of bikes it stocks on any given day which in turn would maximize its profits.

---
## Objective
In this project we will build a simple multi-layer-perceptron (MLP) or a Neural Network (NN) model to predict the number of riders BikeShare would get on a given day. MLP/NN will be build from scratch just using `numpy`. We will not be using any sophisticated deep-learning frameworks such as PyTorch or Tensorflow for building and training the network. The main aim here is to build the network from ground up in order to get the deeper understanding of inner workings of a typical neural network.

---
## Dataset
- Dataset used in this project is sourced from UCI Machine Learning Repository. Data is available as daily (daily.csv) and hourly (hour.csv) basis. 
- For this project we'll just make use of hourly data which is more granular compare to daily data. Dataset provides 2 years worth of data and its relatively small in size, a copy of `hour.csv` is available in the `data` folder in this repo. The complete dataset can be accessed from [here](https://archive.ics.uci.edu/ml/datasets/Bike+Sharing+Dataset)
- Hourly dataset features...
    - instant: record index
    - dteday : date
    - season : season (1:winter, 2:spring, 3:summer, 4:fall)
    - yr : year (0: 2011, 1:2012)
    - mnth : month ( 1 to 12)
    - hr : hour (0 to 23)
    - holiday : weather day is holiday or not
    - weekday : day of the week
    - workingday : if day is neither weekend nor holiday is 1, otherwise is 0.
    + weathersit :
    - 1: Clear, Few clouds, Partly cloudy, Partly cloudy
    - 2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist
    - 3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds
    - 4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog
    - temp : Normalized temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-8, t_max=+39 (only in hourly scale)
    - atemp: Normalized feeling temperature in Celsius. The values are derived via (t-t_min)/(t_max-t_min), t_min=-16, t_max=+50 (only in hourly scale)
    - hum: Normalized humidity. The values are divided to 100 (max)
    - windspeed: Normalized wind speed. The values are divided to 67 (max)
    - casual: count of casual users
    - registered: count of registered users
    - cnt: count of total rental bikes including both casual and registered

Quote from the dataset readme.txt...
> <cite> Bike-sharing rental process is highly correlated to the environmental and seasonal settings. For instance, weather conditions, precipitation, day of week, season, hour of the day, etc. can affect the rental behaviors. The core data set is related to the two-year historical log corresponding to years 2011 and 2012 from Capital Bikeshare system, Washington D.C., USA which is 
publicly available in http://capitalbikeshare.com/system-data. We aggregated the data on two hourly and daily basis and then extracted and added the corresponding weather and seasonal information. Weather information are extracted from http://www.freemeteo.com. </cite>

---

## Solution Approach
- We start with Loading and then exploring the data by looking at available features and any obvious pattern/relationship between the features and target variable (`cnt`)
- One hot encoding is then applied to `categorical` features
- Test set is removed from the dataset and kept aside.
- `mean` and `standard-deviation` is calculated for each of the continuous features in the remaining data (training + validation).
- Both sets above are then standardizing (scaled to std=1, mean=0) by using the statistics calculated above.
- The dataset is then split into training and validation set (60 days worth of data)
- The Network is built as a python class `NeuralNetwork` with just one input, one hidden and one output layer where hidden layer uses a `sigmoid` activation
- The Network trains using Stochastic Gradient Descend (SGD) method where a random batch of data-points are pushed through the netowrk and `network-weights` are updated once for this batch then next random batch is processed. This continues for a given number of epochs (iterations), once done we have a network with updated weights i.e. a trained network that can be used by BikeShare for making prediction. 

### Loss function and Gradient Descend 
For each data-point the output of a layer is computed as...

$$ \Large{\hat y = \sum_{j}^{m}{w_j \cdot x}}\tag{1}$$

where...<br> $\hat y$ is the `predicted-output`, $m$ is the `number of nodes` in the layer, $j$ is the `node index`and $x$ is the `input data-point`

Note that, in practice the above computation is implemented as the `dot product` or `matrix-multiplication` of input ($X$) and weight ($W$) matrices <br>

We are using `MSE` (Mean Squared Error) as the loss function. Once `network-error` (E) is calculated its propagated back through the network by using `gradient-descend`

$$\Large{E = \frac{1}{2}(y - \hat y)^2}\tag{2}$$

where...<br> $m$ is the `number of nodes`, $j$ is the `node index`and $y$ is actual target

Now in order to propagate the MSE error (E) back and update the network weights we need to know the incremental-quantity (delta-weight) by which network-weights to be updated. This quantity is nothing but a fraction of `negative of the gradient` of the error function (MSE in this case). In practice, the delta weight is calculated for each of the network-weights by taking the `partial-derivative` of the error-function (E above) with respect to the respective weight, scaled by the `learning-rate` hyperparameter. Its then used to update the network-weight.

$$\Large{\Delta w = - \eta \cdot \frac{\partial E}{\partial w}}\tag{3}$$

The partial derivative of error-function E w.r.t weights would result in below equation...

$$\Large{\Delta w = - (y - \hat y) \cdot f^\prime(h) \cdot x}\tag{4}$$

where...<br>$\Delta w$ is the `delta weight`,  $w$ is `a network weight` and $x$ is the input. $f^\prime$ is the derivative of $f$ which is the activation function used in the layer and $h$ is linear output ($w \cdot x $) of the layer.

For implementation convenience we compute an `error-term` $\delta$ as... 
$$\Large{\delta = (y - \hat y) \cdot f^\prime(h)}\tag{5}$$

With this, our equation (4) above can be re-written as...
$$\Large{\Delta w = - \delta \cdot x}\tag{6}$$

<br>Finally the network weight will be updated as...

$$\Large{w = w + } \eta \cdot \large{\Delta}\Large{w}\tag{7}$$

where...
$\eta$ is the `learning-rate`

### Implementation details

> NeuralNetwork class implements `train` method, where...
>    - It gets a batch of data-points (sample/record). It then pushes each data-point forward using `forward_pass_train` method and computes the final output for this data point. 
>    - It then computes the `delta-weight` (a small change in network weights) for each data-point using `backpropagation` method. SGD algorithm is used to compute delta-weight.
>    - forward-pass and backward-pass is performed repeatedly for each data-point in the given batch until all data-points are processed.
>    - In the end we'll have cumulative `delta-weights` for the whole batch 
>    - Then `update_weights` method is invoked once per batch. It first computes the average delta-weights for the batch from cumulative delta-weights, it then updates `network-weights` by delta-weights scaled by the `learning-rate` hyperparameter  
> NeuralNetwork class implements `forward_pass_train` method, where...
>    - It produces the final output $\hat y$ for each data-point as per equation (1)

> NeuralNetwork class implements `backpropagation` method, where...
>    - It Computes the `final output error` as shown below 
> $$ \Large{output\_error = y - \hat y}$$
> <br><br> It then computes the output `error-term` as per equation (5)...
> $$\Large{\delta_o = output\_error \cdot f^\prime(o) = output\_error \cdot 1}$$
> <br>Note that $f(o)$ is the output of final layer, since there is no activation function in output layer (i.e. its just a linear function) its derivative $f^\prime(o)$ is 1
> It then computes and accumulates (for each data-point processed) the delta-weight between `hidden` (h) and `output` (o) layer as per equation (6)...
>
> $$\Large{\Delta w_{h\_o} = \Delta w_{h\_o} + \delta_o \cdot f(h)}$$
> where...
> $f(h)$ is the output of hidden layer
>
> It then computes the `hidden-error` (error contribution of hidden layer) as a fraction of of output-error-term proportional to weights between hidden and output layer
>
> $$\Large{hidden\_error = \delta_o \cdot w_{h\_o}}$$
> where...
> $w_{h\_o}$ is the weight between hidden and output layer
>
> `hidden-error-term` is computed as per equation (5)...
>
> $$\Large{\delta_h = hidden\_error \cdot f^\prime(h) = output\_error \cdot f(h) \cdot (1 - f(h))}$$
> <br>Note that $f(h)$ is the output of final layer, since `sigmoid` activation function is used in hidden layer its derivative $f^\prime(h)$ is $f(h) \cdot (1 - f(h))$
>
> It then computes and accumulates (for each data-point processed) the delta-weight between `input` (x) and `hidden` (h) layer as per equation (6)...
>
> $$\Large{\Delta w_{x\_h} = \Delta w_{x\_h} + \delta_h \cdot x}$$

> NeuralNetwork class implements `update_weights` method, where...
>    - It first calculates the average delta-weights (note that in above calculation the delta-weights is calculated a cumulative sum across all data point) 
>    - Finally it updates the network-weights as per equation (7)...
>
>$$\Large{w_{h\_o} = w_{h\_o} + \eta \cdot \Delta w_{h\_o}}$$
>
>$$\Large{w_{x\_h} = w_{x\_h} + \eta \cdot \Delta w_{x\_h}}$$

> NeuralNetwork class implements `run` method, where...
>    - It gets a batch of data-points. It then pushes the data batch forward and returns prediction (number of expected riders) for each data-point in the batch. 
>    - Run method can be used for final prediction using the trained model

---

## Evaluation Criteria
* `MSE` is used as primary metric for loss calculation and model evaluation, formula shown in equation (2) above
* Our goal is to obtain the lowest possible MSE for predictions made using the separately kept `test` dataset

---
## How To Use
1. Ensure below listed packages are installed
    - `numpy`
    - `pandas`
    - `matplotlib`
2. Download `Predicting_bike_sharing_users.ipynb` jupyter notebook from this repo
3. Download the dataset (hour.csv) from the repo and place it in `data` sub-folder 
4. Run the notebook from start to finish. You may want to experiment with the network `hyperparameters` to get even better predictions. Note that data provided is only for 2 years, network seems to be making mistakes during holiday seasons. Possibly, use oversampling techniques to create synthetic data for holiday periods and re-train the network again to see if its able to model the holiday season anomaly better. 
5. Predict using trained model...
    ```python
    preds = network.run(data_points)
    ```
---

## Credits
- Title photo by <a href="https://unsplash.com/@markuswinkler?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Markus Winkler</a> on <a href="https://unsplash.com/s/photos/bike-rental?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText">Unsplash</a>
---

## License

MIT License

Copyright (c) [2021] [Sunil S. Singh]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.

---

## Author Info

- Twitter - [@sunilssingh6](https://twitter.com/sunilssingh6)
- Linkedin - [Sunil S. Singh](https://linkedin.com/in/sssingh)

[Back To The Top](#Credit-Card-Fraud-Detection)

---
