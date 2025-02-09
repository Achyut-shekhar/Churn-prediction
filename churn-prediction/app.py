# coding: utf-8

import pandas as pd
from flask import Flask, request, render_template
import pickle

app = Flask("__name__")

# Load the initial dataset
df_1 = pd.read_csv("tel_churn.csv")

@app.route("/")
def loadPage():
    return render_template('home.html', query="")


@app.route("/", methods=['POST'])
def predict():
    # Collect input data from the form
    inputQuery1 = request.form['query1']  # SeniorCitizen
    inputQuery2 = request.form['query2']  # MonthlyCharges
    inputQuery3 = request.form['query3']  # TotalCharges
    inputQuery4 = request.form['query4']  # gender
    inputQuery5 = request.form['query5']  # Partner
    inputQuery6 = request.form['query6']  # Dependents
    inputQuery7 = request.form['query7']  # PhoneService
    inputQuery8 = request.form['query8']  # MultipleLines
    inputQuery9 = request.form['query9']  # InternetService
    inputQuery10 = request.form['query10']  # OnlineSecurity
    inputQuery11 = request.form['query11']  # OnlineBackup
    inputQuery12 = request.form['query12']  # DeviceProtection
    inputQuery13 = request.form['query13']  # TechSupport
    inputQuery14 = request.form['query14']  # StreamingTV
    inputQuery15 = request.form['query15']  # StreamingMovies
    inputQuery16 = request.form['query16']  # Contract
    inputQuery17 = request.form['query17']  # PaperlessBilling
    inputQuery18 = request.form['query18']  # PaymentMethod
    inputQuery19 = request.form['query19']  # tenure

    # Load the trained model
    try:
        model = pickle.load(open("model.sav", "rb"))
    except Exception as e:
        return render_template('home.html', output1="Error loading model: " + str(e), output2="", query="")

    # Create a DataFrame for the new input data
    data = [[inputQuery1, inputQuery2, inputQuery3, inputQuery4, inputQuery5, inputQuery6, inputQuery7, 
             inputQuery8, inputQuery9, inputQuery10, inputQuery11, inputQuery12, inputQuery13, inputQuery14,
             inputQuery15, inputQuery16, inputQuery17, inputQuery18, inputQuery19]]

    new_df = pd.DataFrame(data, columns=['SeniorCitizen', 'MonthlyCharges', 'TotalCharges', 'gender', 
                                         'Partner', 'Dependents', 'PhoneService', 'MultipleLines', 'InternetService',
                                         'OnlineSecurity', 'OnlineBackup', 'DeviceProtection', 'TechSupport',
                                         'StreamingTV', 'StreamingMovies', 'Contract', 'PaperlessBilling',
                                         'PaymentMethod', 'tenure'])

    # Convert tenure to numeric and handle NaN values
    new_df['tenure'] = pd.to_numeric(new_df['tenure'], errors='coerce').fillna(0).astype(int)

    # Group the tenure in bins of 12 months
    labels = ["{0} - {1}".format(i, i + 11) for i in range(1, 72, 12)]
    new_df['tenure_group'] = pd.cut(new_df['tenure'], range(1, 80, 12), right=False, labels=labels)

    # Drop the original tenure column
    new_df.drop(columns=['tenure'], axis=1, inplace=True)

    # Concatenate the new data with the existing dataset
    df_2 = pd.concat([df_1, new_df], ignore_index=True)

    # Create dummy variables for the new input data only
    new_df_dummies = pd.get_dummies(new_df[['gender', 'SeniorCitizen', 'Partner', 'Dependents', 'PhoneService',
                                              'MultipleLines', 'InternetService', 'OnlineSecurity', 'OnlineBackup',
                                              'DeviceProtection', 'TechSupport', 'StreamingTV', 'StreamingMovies',
                                              'Contract', 'PaperlessBilling', 'PaymentMethod', 'tenure_group']])

    # Ensure the columns match the model's expected input
    try:
        new_df_dummies = new_df_dummies.reindex(columns=model.feature_names_in_, fill_value=0)
    except ValueError as e:
        return render_template('home.html', output1="Error in reindexing : " + str(e), output2="", query="")

    # Make predictions
    single = model.predict(new_df_dummies.tail(1))
    probability = model.predict_proba(new_df_dummies.tail(1))[:, 1]

    if single[0] == 1:
        o1 = "This customer is likely to be churned!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)
    else:
        o1 = "This customer is likely to continue!!"
        o2 = "Confidence: {:.2f}%".format(probability[0] * 100)

    return render_template('home.html', output1=o1, output2=o2, 
                           query1=request.form['query1'], 
                           query2=request.form['query2'],
                           query3=request.form['query3'],
                           query4=request.form['query4'],
                           query5=request.form['query5'], 
                           query6=request.form['query6'], 
                           query7=request.form['query7'], 
                           query8=request.form['query8'], 
                           query9=request.form['query9'], 
                           query10=request.form['query10'], 
                           query11=request.form['query11'], 
                           query12=request.form['query12'], 
                           query13=request.form['query13'], 
                           query14=request.form['query14'], 
                           query15=request.form['query15'], 
                           query16=request.form['query16'], 
                           query17=request.form['query17'],
                           query18=request.form['query18'], 
                           query19=request.form['query19'])

app.run(debug=True)