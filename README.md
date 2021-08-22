# Kaggle Data Science Challenge -- MLB player digital engagement forecast

## Basic Introduction
Basically, it is a regression problem. Using all kinds of information of players today,try to predict 4 digital engagements for each player in the next day.

Original Address:`https://www.kaggle.com/c/mlb-player-digital-engagement-forecasting`

## Data
From my view, the most difficult part in this task is the process of data. the challenge provides large amounts of data from different dimensions. For example, the player's static property,like his name,his birth place and so on. Besides, it also provides players daily data, like whether he won the games in that day, how many scores he got and so on. 

Some useful data it provides is stored in json, so I have a script called `json_load.py` to deal with this.


## Before Modeling
After I successfully combine all information into 1 dataframe and create some useful features, there are still some necessary things need to be done.
### 1.filter out the feature with low variance
because it contributes nothing to the explanation of change of targets.
### 2.normalization
it can help the training of model more smoothly.

## Model
For this task, I tried 2 different models, one is the simple neural network(MLP), the other one is the GBDT. And the performances of 2 models on test set are shown below.

**MLP:average test loss:61.72892190670145**

**GBDT:average test loss:56.72765506550651**

## Try yourself
you can run the code by enter the following command in terminal(change the parameters as you want,e.g. use **gbdt**)
`python -W ignore main.py --data_path './data/train_updated.csv' --train_size 150000 --val_size 30000 --model 'ann'`

In summary, this repository only shows you a basic soolution to this data challenge, there are still lots of things can be done to improve the performance, for example, better **feature engineering**, better model parameters setting using methods like **grid search** or **cross validation**. what's more, you can also try more complex models and some **ensemble learning** algorithms to get better performance.



