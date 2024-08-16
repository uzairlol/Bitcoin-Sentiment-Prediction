# Bitcoin Sentiment and Price Prediction

This repository contains a comprehensive machine learning project designed to predict Bitcoin's next-day price movements by combining historical market data with sentiment analysis derived from Wikipedia edits. The approach leverages both financial and sentiment data to enhance predictive accuracy, using advanced machine learning algorithms.

## Overview

Cryptocurrency markets are highly volatile, and traditional price prediction models often fail to account for the impact of public sentiment on market movements. In this project, sentiment data from Wikipedia edits is used to analyze market mood, which is then integrated with historical price data to improve the model's performance. The primary goal is to create a more accurate model that can be used for decision-making in cryptocurrency trading.

## Project Structure

- **`sentimentlive.ipynb`**  
  This notebook focuses on real-time sentiment analysis. It processes Wikipedia edit data to generate sentiment scores, which are then used as input features for the prediction model. Key steps include data acquisition, preprocessing, feature extraction, and sentiment score calculation.

- **`predictionlive.ipynb`**  
  This notebook handles the prediction of Bitcoin's next-day price movement. Using the historical price data combined with the sentiment scores, machine learning algorithms like XGBoost and Random Forest are employed to predict whether Bitcoin's price will rise or fall the next day. It includes data preprocessing, model training, hyperparameter tuning, and evaluation.

## Key Features

### 1. **Sentiment Analysis**
   - **Source**: Wikipedia edits
   - **Processing**: The notebook processes the sentiment data, assigning sentiment scores that reflect the market's mood based on real-time updates.
   - **Integration**: These sentiment scores are added to the dataset and used as features for the price prediction model.

### 2. **Price Prediction**
   - **Data**: Historical Bitcoin price data is fetched using the `yfinance` library. Data includes open, close, high, low prices, volume, and other relevant features.
   - **Algorithms**: 
     - **Random Forest**: A flexible, easy-to-use algorithm that provides robust results.
     - **XGBoost**: An optimized gradient boosting algorithm known for high performance in classification tasks.
   - **Hyperparameter Tuning**: GridSearchCV is used for tuning hyperparameters to ensure the model is optimized for better performance.

## Installation and Requirements

To replicate this project, clone the repository and install the necessary dependencies. You can install them via `pip` using the command:

```bash
pip install -r requirements.txt
```

### Requirements

- Python 3.x
- Jupyter Notebook
- Pandas
- Scikit-learn
- XGBoost
- yfinance
- Joblib
- Matplotlib
- Numpy

## Usage Instructions

### 1. Sentiment Analysis
- Navigate to the `sentimentlive.ipynb` notebook.
- Follow the instructions within the notebook to fetch and process real-time sentiment data.
- The sentiment scores will be calculated and stored, ready to be used in the prediction model.

### 2. Price Prediction
- Navigate to the `predictionlive.ipynb` notebook.
- This notebook uses the historical price data and sentiment scores to train the model.
- The model will predict whether Bitcoin's price will rise or fall on the next day.
- The notebook includes detailed steps for data preprocessing, model training, and prediction.

## Model Explanation

### Random Forest
Random Forest is an ensemble learning method that operates by constructing multiple decision trees during training. It outputs the mode of the classes for classification problems. In this project, Random Forest is used for its robustness and ability to handle noisy data, which is common in financial datasets.

### XGBoost
XGBoost is an implementation of gradient boosted decision trees designed for speed and performance. It has become popular in machine learning competitions and is used in this project for its strong predictive power and ability to model complex relationships in data.

### Hyperparameter Tuning
We used `GridSearchCV` to perform hyperparameter tuning on the models, testing various combinations of hyperparameters to find the best configuration that maximizes accuracy and reduces overfitting.

## Results and Evaluation

The project evaluates the model's performance using various metrics such as precision, accuracy, and recall. Cross-validation is applied during the training phase to ensure the model generalizes well to unseen data.

- **Precision**: Measures the proportion of true positives among the predicted positives.
- **Accuracy**: Indicates the overall correctness of the model's predictions.
- **Recall**: Evaluates how well the model captures all relevant instances.

After training, the model with the best parameters is saved using `joblib` for future use.

## Future Work

- **Feature Engineering**: Explore additional features such as social media sentiment, global economic indicators, and news articles to improve model accuracy.
- **Model Optimization**: Experiment with advanced models like LSTM (Long Short-Term Memory) for time series data.
- **Real-time Prediction**: Develop a pipeline to integrate real-time market and sentiment data for continuous price prediction updates.

## Contributing

Feel free to fork this repository and submit pull requests. Contributions that enhance the model's performance or expand its capabilities are welcome!
