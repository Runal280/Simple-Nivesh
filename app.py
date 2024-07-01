from flask import Flask, render_template
import matplotlib.pyplot as plt
import io
import base64

app = Flask(__name__)

import yfinance as yf
import numpy as np
import pandas as pd
import tensorflow as tf
import requests
from datetime import datetime 
from nltk.sentiment.vader import SentimentIntensityAnalyzer
from tensorflow.keras.layers import Dense, Dropout, LSTM
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

start_date = '2020-01-15'
end_date = datetime.now().strftime('%Y-%m-%d')  # Get current date and time
symbol = 'TATAMOTORS.NS'

df = yf.download(symbol, start=start_date, end=end_date)
csv_file_path = "TATAMOTORS.csv"
df.to_csv(csv_file_path)
df = pd.read_csv("TATAMOTORS.csv")
print(f"Data has been saved to: {csv_file_path}")
print(f"Download your file here: file://{csv_file_path}")
df
df.head()
df.info()
df['Open'].plot(figsize=(13, 6))
plt.xlabel('Time')
df.rolling(7).mean().head(20)
plt.ylabel('Daily Opening Price')
df['Close: 30 Day Mean'] = df['Close'].rolling(30).mean()
df[['Close', 'Close: 30 Day Mean']].plot(figsize=(13, 3))
df['Close'].expanding(min_periods=1).mean().plot(figsize=(13, 3))
training_set = df['Open']
training_set = pd.DataFrame(training_set)

# Data Cleaning
df.isna().any()
# Splitting data into training and testing sets
df_training = pd.DataFrame(df['Close'][:int(len(df) * 0.70)])
df_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])
# Feature scaling
from sklearn.preprocessing import MinMaxScaler
sc = MinMaxScaler(feature_range=(0, 1))
data_training_scaled = sc.fit_transform(df_training)
data_testing_scaled = sc.fit_transform(df_testing)
data_training_scaled

data_training_scaled.shape
# Creating a data structure with 60 timesteps and 1 output
x_train = []
y_train = []
for i in range(60, data_training_scaled.shape[0]):
    x_train.append(data_training_scaled[i - 60:i, 0])
    y_train.append(data_training_scaled[i, 0])

x_train, y_train = np.array(x_train), np.array(y_train)
# Feature engineering
df["Close_Diff"] = df["Close"].diff()  # Daily price difference
df["Target"] = np.where(df["Close_Diff"] > 0, 1, 0)  # Binary target
df.plot()
# Select features and target
features = ["Close", "Volume"]  
target = "Target"

X = df[features]
y = df[target]

import tensorflow as tf
from tensorflow.compat.v1.losses import sparse_softmax_cross_entropy
from keras.layers import Dense, Dropout, LSTM
from keras.models import Sequential

# Building the LSTM model

regressor = Sequential()
# Adding 1st LSTM layer with dropout regularization
regressor.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1], 1)))
regressor.add(Dropout(0.2))
# Adding 2nd LSTM Layer and some dropout regularization
regressor.add(LSTM(units=68, return_sequences=True))
regressor.add(Dropout(0.3))

# Adding 3rd LSTM Layer and some dropout regularization
regressor.add(LSTM(units=80, return_sequences=True))
regressor.add(Dropout(0.4))

# Adding 4th LSTM Layer and some dropout regularization
regressor.add(LSTM(units=80))
regressor.add(Dropout(0.5))
# Building output layer
regressor.add(Dense(units=1))

# Compiling the RNN
regressor.compile(optimizer='adam', loss='mean_squared_error')


# Fitting the RNN to the training set
regressor.fit(x_train, y_train, epochs=5, batch_size=32)
regressor.save('keras_model_LSTM_on_TATA.h5')

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)
df_testing.head()
df_training.head()
df_training.tail()
past_100_days=df_training.tail(100)
final_df=past_100_days.append(df_testing, ignore_index=True)
final_df
input_data=sc.fit_transform(final_df)
input_data
input_data.shape
x_test=[]
y_test=[]
for i in range(100,input_data.shape[0]):
    x_test.append(input_data[i-60:i])
    y_test.append(input_data[i,0])
x_test,y_test=np. array(x_test),np. array(y_test)
print(x_test.shape)
print(y_test.shape)
y_predicted=regressor.predict(x_test)
y_predicted.shape
y_test.shape
y_predicted_original_scale=sc.inverse_transform(y_predicted)
y_test_original_scale = sc.inverse_transform(y_test.reshape(-1,1))


# Load NLP model
nlp_model = SentimentIntensityAnalyzer()
# API Key for newsapi.org
api_key = "57744bb4f5d84228a3eaef4d8ccccfa2"

# Function to get news sentiment using newsapi.org
def get_news_sentiment(symbol):
    url = f"https://newsapi.org/v2/everything?q={symbol}&apiKey={api_key}"
    response = requests.get(url)
    data = response.json()
    articles = data.get('articles', [])
    if articles:
        article_text = articles[0]['content']
        sentiment_score = nlp_model.polarity_scores(article_text)["compound"]
        return sentiment_score
    else:
        return None

# Make predictions on the test set
predictions = model.predict(X_test)

current_close_price = 31.65
current_volume = 10116229
current_data = pd.DataFrame({"Close": [current_close_price], "Volume": [current_volume]})
buying_probability = model.predict_proba(current_data)[0, 1]

# Calculate buy and sell percentages
buy_percentage = buying_probability * 100
sell_percentage = (1 - buying_probability) * 100

# Sentiment analysis
sentiment_score = get_news_sentiment('TATAMOTORS')

plt.figure(figsize=(12,6))
plt.plot(y_test,'b',label='original price')
plt.plot(y_predicted,'r',label='predicted price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
plt.show()

@app.route('/')
def index():
    # Calculate sentiment score, buy percentage, and sell percentage
    sentiment_score = get_news_sentiment('TATAMOTORS')
    # Plot a pie chart
    labels = ["Sell", "Buy"]
    sizes = [sell_percentage, buy_percentage]
    colors = ["red", "green"]

    fig, ax = plt.subplots(figsize=(6, 6))  
    ax.pie(sizes, labels=labels, colors=colors, autopct="%1.1f%%", startangle=90)
    ax.axis("equal")
    ax.set_title("Buying Probability")
    
    # Convert plot to PNG image
    img = io.BytesIO()
    plt.savefig(img, format='png')
    img.seek(0)
    
    # Encode PNG image to base64
    chart_url = base64.b64encode(img.getvalue()).decode()
    plt.close(fig)
    
    # Render the template with the variables
    return render_template('index.html', chart_url=chart_url, sentiment_score=sentiment_score, buy_percentage=buy_percentage, sell_percentage=sell_percentage)

if __name__ == '__main__':
    app.run(debug=True)
