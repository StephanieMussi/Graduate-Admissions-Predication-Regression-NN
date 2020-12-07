# Graduate_Admissions_Predication_Regression_NN
The [Graduate Admission Prediction dataset] (https://www.kaggle.com/mohansacharya/graduate-admissions) contains information like GRE scores, recommendation letter and a probability of admission between 0 and 1. The aim of this project is to perform regression task on this dataset, more specifically, to implement neural network models that take in variables and output the admission probability.  

## 3-Layer Model
The 3-layer neural network includes a hidden layer of 10 neurons. The linear activation function is used for output layer.  
```python
no_neurons = 10
model = Sequential([Dense(no_neurons, activation='relu', 
                          kernel_initializer=RandomUniform(w_min_relu, w_max_relu), 
                          kernel_regularizer=l2(beta)),
                    Dense(1, activation = 'linear',
                          kernel_initializer=RandomUniform(w_min_linear, w_max_linear),
                          kernel_regularizer=l2(beta))])
```
The model is trained with SGD optimizer for 500 epochs, and the loss curve is plotted. As it is observed, the loss converges after around 100 epochs. 
<img src="https://github.com/StephanieMussi/Graduate_Admissions_Predication_Regression_NN/blob/main/Figures/3Loss.png" width="300" height="200">  

In order to have a visual perception of the prediction accuracy, the target value and prediction value of 50 samples are plotted:  
<img src="https://github.com/StephanieMussi/Graduate_Admissions_Predication_Regression_NN/blob/main/Figures/Target&Prediction.png" width="300" height="200">   

## 3-Layer Model With RFE
Recursive feature elimination removes unnecessary features which cause maximum drop (minimum increase) in loss.  
For comparison, the model with full set of inputs are trained, and the mean squared error is recorded.  

### Remove 1 feature
To determine which input to remove, the model is trained without _1st/2nd/.../7th_ input variable. The the inputs are labeled from left to right starting from the second column in ["admission_prediction.csv"](https://github.com/StephanieMussi/Graduate_Admissions_Predication_Regression_NN/blob/main/admission_predict.csv).   
The losses are as below:  
| Remove which input | 1st | 2nd | 3rd | 4th | 5th | 6th | 7th |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Loss | 0.0071 |  0.0059 | 0.0058 | 0.0064 | 0.0051 | 0.0067 |  0.0065 |
As observed, removing 5th input leads to minimum loss.  

### Remove 2 features 
Similarly, after the 5th input is removed, the model is trained without _1st/2nd/3rd/4th/6th/7th_ input. The losses are summarized in the table: 
| Remove which input | 1st | 2nd | 3rd | 4th | 6th | 7th |
| :-: | :-: | :-: | :-: | :-: | :-: | :-: |
| Loss | 0.0057 |  0.0070 | 0.0054 | 0.0053 | 0.0071 | 0.0056 |  

In this way, the 3rd input is eliminated.   
  
The losses of using 7, 6 and 5 inputs are as below:  
| Number of inputs | 7 | 6 | 5 |
| :-: | :-: | :-: | :-: |
| Loss | 0.0065 | 0.0052 | 0.0054 |  
As observed, removing one inputs improve the performance while removing two inputs does the opposite. Therefore, RFE needs to stop at 6 inputs.  
The graph of losses is plotted:  
<img src="https://github.com/StephanieMussi/Graduate_Admissions_Predication_Regression_NN/blob/main/Figures/3RFELoss.png" width="300" height="200">   
It can be seen that the model with 6 inputs converges the fastest, which is coherent with the statement above.  


## Comparison with 4- and 5-Layer Models
