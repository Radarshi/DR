import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Load your dataset
data = pd.read_csv('data.csv')  # Assuming you have a CSV file

# Sample data preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters
    text = re.sub(r'\@\w+|\#','', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Apply preprocessing
data['clean_text'] = data['text'].apply(preprocess_text)

#Feature Extraction 
from sklearn.feature_extraction.text import TfidfVectorizer

# Vectorize text data
tfidf = TfidfVectorizer(max_features=5000)
X = tfidf.fit_transform(data['clean_text']).toarray()
y = data['severity_level']  # Assuming this is your target variable

#Model Building 
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build the neural network model
model = Sequential()
model.add(Dense(512, input_dim=X_train.shape[1], activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(5, activation='softmax'))  # 5 classes for severity levels

model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])

# Train the model
model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

#Model Evaluation 
from sklearn.metrics import classification_report, accuracy_score

# Predict on the test set
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)

# Evaluation
print("Accuracy:", accuracy_score(y_test, y_pred_classes))
print(classification_report(y_test, y_pred_classes))

#Flask and API Endpoint 
from flask import Flask, request, jsonify

app = Flask(__name__)

@app.route('/predict', methods=['POST'])
def predict_severity():
    data = request.get_json(force=True)
    text = preprocess_text(data['text'])
    vector = tfidf.transform([text]).toarray()
    prediction = model.predict(vector)
    severity = prediction.argmax(axis=1)[0]  # Get predicted class
    return jsonify({'severity_level': int(severity)})

if __name__ == '__main__':
    app.run(debug=True)



