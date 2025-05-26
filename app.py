from flask import Flask, request, render_template
import pandas as pd
from sklearn.tree import DecisionTreeClassifier
import re

app = Flask(__name__)

# Load training data
df = pd.read_csv('training.csv')
X = df.drop(['Disease', 'Medication'], axis=1)
y = df['Disease']
med_map = df[['Disease', 'Medication']].drop_duplicates().set_index('Disease')['Medication'].to_dict()

# Train model
clf = DecisionTreeClassifier()
clf.fit(X, y)

# List of known symptoms from dataset
known_symptoms = X.columns.tolist()

def extract_symptoms(text):
    text = text.lower()
    detected = []
    for symptom in known_symptoms:
        if re.search(r'\b' + re.escape(symptom.lower()) + r'\b', text):
            detected.append(symptom)
    return detected

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    user_input = request.form['user_input']
    symptoms = extract_symptoms(user_input)

    if not symptoms:
        return render_template('index.html', prediction="No recognizable symptoms found.", symptoms="", medication="")

    # Build input vector
    input_vector = [1 if s in symptoms else 0 for s in known_symptoms]
    prediction = clf.predict([input_vector])[0]
    medication = med_map.get(prediction, "No suggestion available")

    return render_template('index.html',
                           prediction=prediction,
                           symptoms=", ".join(symptoms),
                           medication=medication)

if __name__ == '__main__':
    app.run(debug=True)
