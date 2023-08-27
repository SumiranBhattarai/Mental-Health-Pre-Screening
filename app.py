import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder
from flask import Flask, render_template, request
from sklearn.preprocessing import LabelEncoder, MinMaxScaler
from flask import redirect, url_for
app = Flask(__name__)


# ... (other imports and app setup)

# Load the pre-trained model
with open('model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the dataset or create your dataset dictionary
# dataset = ..
dataset = pd.read_csv('cleaned.csv')

# Perform label encoding
label_encoders = {}
label_mapping = {}
for feature in dataset:
    le = LabelEncoder()
    le.fit(dataset[feature])
    label_encoders[feature] = le
    label_mapping[feature] = dict(zip(le.transform(le.classes_), le.classes_))
    dataset[feature] = le.transform(dataset[feature])

# Scaling Age
scaler = MinMaxScaler()
dataset['Age'] = scaler.fit_transform(dataset[['Age']])


@app.route('/', methods=['GET', 'POST'])
def survey():
    prediction_result = None
    if request.method == 'POST':
        # Retrieve form data
        age = float(request.form['age'])
        gender = request.form['gender']
        family_history = request.form['family-history']
        benefits = request.form['mental-health-benefits']
        care_options = request.form['care-options']
        anonymity = request.form['talk-to-coworker']
        leave = request.form['medical-leave-ease']
        work_interfere = request.form['interference-with-work']

        # Process the data or save to a database

        # Preprocess the data for model prediction
        gender_encoded = label_encoders['Gender'].transform([gender])[0]
        family_history_encoded = label_encoders['family_history'].transform([
                                                                            family_history])[0]
        benefits_encoded = label_encoders['benefits'].transform([benefits])[0]
        care_options_encoded = label_encoders['care_options'].transform([care_options])[
            0]
        anonymity_encoded = label_encoders['anonymity'].transform([anonymity])[
            0]
        leave_encoded = label_encoders['leave'].transform([leave])[0]
        work_interfere_encoded = label_encoders['work_interfere'].transform([
                                                                            work_interfere])[0]

        # Convert data to a list of feature values in the same order as the model's input
        feature_values = [age, gender_encoded, family_history_encoded, benefits_encoded,
                          care_options_encoded, anonymity_encoded, leave_encoded, work_interfere_encoded]

        # Use the loaded model to predict the categorical variable
        prediction = model.predict([feature_values])

        # Reverse transform the prediction to get the original categorical label
        print(prediction)

        Yes = "You need to check with a doctor for proper diagnosis and treatment"
        No = "You are <h6>fine</h6>, but you can still check with a doctor for proper diagnosis and treatment"
        # Do something with the prediction (e.g., display it in the template)
        prediction_result = "<h3>You need to check with a doctor for proper diagnosis and treatment</h3>" if prediction == 1 else "<h3>You are fine, but you can still check with a doctor for proper diagnosis and treatment</h3>"

    return render_template('survey.html', prediction_result=prediction_result)

# ... (rest of the code)


if __name__ == '__main__':
    app.run(debug=True)
