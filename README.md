# Back-Propagation-Neural-Network

[![BPNN](https://badgen.net/badge/github/BPNN?icon&label=GitHub)](https://github.com/GavinTechStudio/Back-Propagation-Neural-Network) [![C++](https://img.shields.io/badge/support-C%2B%2B11%20or%20later-blue?style=flat&logo=cplusplus)](https://github.com/topics/cpp) [![CMake](https://img.shields.io/badge/support-v2.8.12%20or%20later-blue?style=flat&logo=cmake)](https://cmake.org/) [![update](https://img.shields.io/github/last-commit/GavinTechStudio/Back-Propagation-Neural-Network)](https://github.com/GavinTechStudio/Back-Propagation-Neural-Network/commits) [![pages-build-deployment](https://github.com/GavinTechStudio/Back-Propagation-Neural-Network/actions/workflows/pages/pages-build-deployment/badge.svg)](https://github.com/GavinTechStudio/Back-Propagation-Neural-Network/actions/workflows/pages/pages-build-deployment) 

[This project](https://github.com/GavinTechStudio/Back-Propagation-Neural-Network)It is for the project [**GavinTechStudio/bpnn_with_cpp**](https://github.com/GavinTechStudio/bpnn_with_cpp) The code reconstruction and the implementation of basic BP neural network based on C++ will help to deeply understand the principles of BP neural network.

## Project Structure

```
.
├── CMakeLists.txt
├── data
│   ├── testdata.txt
│   └── traindata.txt
├── lib
│   ├── Config.h
│   ├── Net.cpp
│   ├── Net.h
│   ├── Utils.cpp
│   └── Utils.h
└── main.cpp
```

#### Main framework

- Net：Network specific implementation
- Config：Network parameter settings
- Utils：Tools
  - Data loading
  - Activation Function
- main：Network specific applications

## Training Principles

> For the specific formula derivation, please watch the video explanation [Completely understand the BP neural network theoretical derivation + code implementation（C++）](https://www.bilibili.com/video/BV1Y64y1z7jM?p=1)

#### Note: This part of the document contains a lot of mathematical formulas. Since GitHub markdown does not support mathematical formula rendering, the following reading method is recommended：

1. If you are using Chrome, Edge or Firefox You can install plugins for browsers like[MathJax Plugin for Github](https://chrome.google.com/webstore/detail/mathjax-plugin-for-github/ioemnmodlmafdkllaclgeombjnmnbima)（Requires network access chrome web store）。
2. Use [**PDF**](https://gavintechstudio.github.io/Back-Propagation-Neural-Network/README.pdf) way to read.
2. Use [**Pre-rendered static web pages**](https://gavintechstudio.github.io/Back-Propagation-Neural-Network/README.html) Do some reading (**recommended**).
2. Press the `.` key or [Click on the link](https://github.dev/GavinTechStudio/Back-Propagation-Neural-Network) Enter GitHub online IDE to preview the `README.md` file.

### 0. Neural network structure diagram

![](img/net-info.png)

### 1. Forward（Forward Propagation）

#### 1.1 Propagate from the input layer to the hidden layer

$$
h_j = \sigma( \sum_i x_i w_{ij} - \beta_j )
$$

Where $h_j$ is the value of the $j$th hidden layer node, $x_i$ is the value of the $i$th input layer node, $w_{ij}$ is the weight from the $i$th input layer node to the $j$th hidden layer node, $\beta_j$ is the bias value of the $j$th hidden layer node, and $\sigma(x)$ is the **Sigmoid** activation function. This expression will continue to be used later. The expression is as follows:

$$
\sigma(x) = \frac{1}{1+e^{-x}}
$$

The code implementation in this project is as follows：

```C++
for (size_t j = 0; j < Config::HIDENODE; ++j) {
	double sum = 0;
    for (size_t i = 0; i < Config::INNODE; ++i) {
        sum += inputLayer[i]->value * 
               inputLayer[i]->weight[j];
    }
    sum -= hiddenLayer[j]->bias;
  
    hiddenLayer[j]->value = Utils::sigmoid(sum);
}
```

#### 1.2 Propagate from hidden layer to output layer

$$
\hat{y_k} = \sigma( \sum_j h_j v_{jk} - \lambda_k )
$$

Where $\hat{y_k}$ is the value (prediction value) of the $k$th output layer node, $h_j$ is the value of the $j$th hidden layer node, $v_{jk}$ is the weight from the $j$th hidden layer node to the $k$th output layer node, $\lambda_k$ is the bias value of the $k$th output layer node, and $\sigma(x)$ is the activation function.

The code implementation in this project is as follows:

```C++
for (size_t k = 0; k < Config::OUTNODE; ++k) {
    double sum = 0;
    for (size_t j = 0; j < Config::HIDENODE; ++j) {
        sum += hiddenLayer[j]->value * 
               hiddenLayer[j]->weight[k];
    }
    sum -= outputLayer[k]->bias;
    
    outputLayer[k]->value = Utils::sigmoid(sum);
}
```

### 2. Calculating the loss function（Loss Function）

The loss function is defined as follows:

$$
Loss = \frac{1}{2}\sum_k ( y_k - \hat{y_k} )^2
$$

Where $y_k$ is the target value (true value) of the $k$th output layer node, and $\hat{y_k}$ is the value (predicted value) of the $k$th output layer node.

The code implementation in this project is as follows:

```C++
double loss = 0.f;

for (size_t k = 0; k < Config::OUTNODE; ++k) {
    double tmp = std::fabs(outputLayer[k]->value - label[k]);
    los += tmp * tmp / 2;
}
```

### 3. Backward（Back Propagation）

Optimization is performed using gradient descent.

#### 3.1 Calculate $\Delta \lambda_k$ (corrected value of the output layer node bias value)

The calculation formula is as follows (when the activation function is Sigmoid):

$$
\Delta \lambda_k = - \eta (y_k - \hat{y_k}) \hat{y_k} (1 - \hat{y_k})
$$

Where $\eta$ is the learning rate (the other variables have already appeared above and are no longer marked).

The code implementation in this project is as follows:

```C++
for (size_t k = 0; k < Config::OUTNODE; ++k) {
    double bias_delta = 
        -(label[k] - outputLayer[k]->value) *
        outputLayer[k]->value *
        (1.0 - outputLayer[k]->value);
    
    outputLayer[k]->bias_delta += bias_delta;
}
```

#### 3.2 Calculate $\Delta v_{jk}$ (corrected value of the weight from the hidden layer node to the output layer node)

The calculation formula is as follows (when the activation function is Sigmoid):

$$
\Delta v_{jk} = \eta ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) h_j
$$

Where $h_j$ is the value of the $j$th hidden layer node (the other variables have appeared above and are no longer marked).

The code implementation in this project is as follows:

```C++
for (size_t j = 0; j < Config::HIDENODE; ++j) {
    for (size_t k = 0; k < Config::OUTNODE; ++k) {
        double weight_delta =
            (label[k] - outputLayer[k]->value) * 
            outputLayer[k]->value * 
            (1.0 - outputLayer[k]->value) * 
            hiddenLayer[j]->value;

		hiddenLayer[j]->weight_delta[k] += weight_delta;
    }
}
```

#### 3.3 Calculate $\Delta \beta_j$ (corrected value of the hidden layer node bias value)

The calculation formula is as follows (when the activation function is Sigmoid):

$$
\Delta \beta_j = - \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) v_{jk} h_j ( 1 - h_j )
$$

Where $v_{jk}$ is the weight from the $j$th hidden layer node to the $k$th output layer node (the other variables have appeared above and are no longer labeled).

The code implementation in this project is as follows:

```C++
for (size_t j = 0; j < Config::HIDENODE; ++j) {
	double bias_delta = 0.f;
	for (size_t k = 0; k < Config::OUTNODE; ++k) {
		bias_delta += 
            -(label[k] - outputLayer[k]->value) * 
            outputLayer[k]->value * 
            (1.0 - outputLayer[k]->value) * 
            hiddenLayer[j]->weight[k];
	}
	bias_delta *= 
        hiddenLayer[j]->value * 
        (1.0 - hiddenLayer[j]->value);

	hiddenLayer[j]->bias_delta += bias_delta;
}
```

#### 3.4 Calculate $\Delta w_{ij}$ (corrected value of the weight from the input layer node to the hidden layer node)

The calculation formula is as follows (when the activation function is Sigmoid):

$$
\Delta w_{ij} = \eta \sum_k ( y_k - \hat{y_k} ) \hat{y_k} ( 1 - \hat{y_k} ) v_{jk} h_j ( 1 - h_j ) x_i
$$

Where $x_i$ is the value of the $i$th input layer node (the remaining variables have appeared above and are not labeled here).

The code implementation in this project is as follows:

```C++
for (size_t i = 0; i < Config::INNODE; ++i) {
	for (size_t j = 0; j < Config::HIDENODE; ++j) {
		double weight_delta = 0.f;
		for (size_t k = 0; k < Config::OUTNODE; ++k) {
			weight_delta +=
                (label[k] - outputLayer[k]->value) * 
                outputLayer[k]->value * 
                (1.0 - outputLayer[k]->value) * 
                hiddenLayer[j]->weight[k];
        }
		weight_delta *=
            hiddenLayer[j]->value * 
            (1.0 - hiddenLayer[j]->value) * 
            inputLayer[i]->value;

		inputLayer[i]->weight_delta[j] += weight_delta;
	}
}
```
