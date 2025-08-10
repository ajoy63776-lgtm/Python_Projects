from flask import Flask, request
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
app = Flask(__name__)
datafream = pd.read_csv(r'C:\Users\AJOY PAUL\Downloads\health_data.csv')


label_encoders = {}
for col in ['Gender', 'Smoking', 'FamilyHistory', 'Disease']:
    le = LabelEncoder()
    datafream[col] = le.fit_transform(datafream[col])
    label_encoders[col] = le

X = datafream.drop('Disease', axis=1)
y = datafream['Disease']
model = RandomForestClassifier()
model.fit(X, y)
@app.route('/')
def form():
    return '''
    <h2>Health Prediction</h2>
    <form action="/predict" method="post">
        Age: <input type="text" name="Age"><br>
        Gender (Male/Female): <input type="text" name="Gender"><br>
        BMI: <input type="text" name="BMI"><br>
        BP: <input type="text" name="BP"><br>
        Sugar: <input type="text" name="Sugar"><br>
        Cholesterol: <input type="text" name="Cholesterol"><br>
        Smoking (Yes/No): <input type="text" name="Smoking"><br>
        Family History (Yes/No): <input type="text" name="FamilyHistory"><br>
        <input type="submit" value="Predict">
    </form>
    '''
@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input_raw = [
            int(request.form['Age']),
            request.form['Gender'],
            float(request.form['BMI']),
            float(request.form['BP']),
            float(request.form['Sugar']),
            float(request.form['Cholesterol']),
            request.form['Smoking'],
            request.form['FamilyHistory']
        ]

        user_input_encoded = [
            user_input_raw[0],
            label_encoders['Gender'].transform([user_input_raw[1]])[0],
            user_input_raw[2],
            user_input_raw[3],
            user_input_raw[4],
            user_input_raw[5],
            label_encoders['Smoking'].transform([user_input_raw[6]])[0],
            label_encoders['FamilyHistory'].transform([user_input_raw[7]])[0]
        ]

        prediction = model.predict([user_input_encoded])
        predicted_label = label_encoders['Disease'].inverse_transform(prediction)[0]

        return f"<h2>Prediction Result: {predicted_label}</h2><a href='/'>Back</a>"

    except Exception as e:
        return f"<h2>Error: {str(e)}</h2><a href='/'>Back</a>"

if __name__ == '__main__':
    app.run(debug=True)

#CLICK CTRL+ PRESS Running on http://127.0.0.1:5000
#Press CTRL+C to quit