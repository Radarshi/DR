import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from sklearn.metrics import classification_report, accuracy_score
from flask import Flask, request, jsonify

# Download NLTK resources
nltk.download('stopwords')
nltk.download('punkt')

# Sample data preprocessing function
def preprocess_text(text):
    # Lowercase
    text = text.lower()
    # Remove URLs
    text = re.sub(r'http\S+|www\S+|https\S+', '', text, flags=re.MULTILINE)
    # Remove special characters and numbers
    text = re.sub(r'\@\w+|\#|\d+', '', text)
    # Tokenize
    words = word_tokenize(text)
    # Remove stopwords
    words = [word for word in words if word not in stopwords.words('english')]
    return ' '.join(words)

# Function to load data from CSV
def load_data(file_path):
    return pd.read_csv(file_path)

# Function to save processed data to CSV
def save_processed_data(data, output_file_path):
    data.to_csv(output_file_path, index=False)

# Function to train a model
def train_model(X, y):
    model = Sequential()
    model.add(Dense(512, input_dim=X.shape[1], activation='relu'))
    model.add(Dense(128, activation='relu'))
    model.add(Dense(5, activation='softmax'))  # 5 classes for severity levels
    model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

# Main function
def main(input_csv_path, output_csv_path):
    # Load dataset
    data = load_data(input_csv_path)

    # Apply preprocessing
    data['clean_text'] = data['text'].apply(preprocess_text)

    # Vectorize text data
    tfidf = TfidfVectorizer(max_features=5000)
    X = tfidf.fit_transform(data['clean_text']).toarray()
    y = data['severity_level']  # Assuming this is your target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = train_model(X_train, y_train)
    model.fit(X_train, y_train, epochs=10, batch_size=64, validation_data=(X_test, y_test))

    # Predict on the test set
    y_pred = model.predict(X_test)
    y_pred_classes = y_pred.argmax(axis=1)

    # Evaluation
    print("Accuracy:", accuracy_score(y_test, y_pred_classes))
    print(classification_report(y_test, y_pred_classes))

    # Save results to a new CSV file
    data['predicted_severity_level'] = model.predict_classes(tfidf.transform(data['clean_text']).toarray())
    save_processed_data(data, output_csv_path)

# Flask and API Endpoint
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
    input_csv_path = 'input_data.csv'  # Path to the input CSV file
    output_csv_path = 'processed_data.csv'  # Path to save the output CSV file
    main(input_csv_path, output_csv_path)
    app.run(debug=True)
