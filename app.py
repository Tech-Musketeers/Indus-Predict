from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import joblib
from sklearn.preprocessing import StandardScaler
import os

app = Flask(_name_)

# Load the model using joblib
model = joblib.load('equip_mnts.pkl')

# Load the data for scaling
df = pd.read_csv(os.path.join(os.path.dirname(_file_), "templates", "machinery_data.csv"))

# Initialize the scaler and fit it with the data
scaler = StandardScaler()
scaler.fit(df[['sensor_1', 'sensor_2', 'sensor_3', 'operational_hours']])

@app.route('/')
def index():
    return render_template("index.html")

@app.route('/reference_data')
def reference_data():
    # Display reference data from the CSV
    reference_table = df.to_html(classes='table table-striped', index=False)
    return render_template("reference_data.html", tables=[reference_table])

@app.route("/feed_data")
def feed_data():
    sensors1 = sorted(df["sensor_1"].unique())
    sensors2 = sorted(df["sensor_2"].unique())
    sensors3 = sorted(df["sensor_3"].unique())
    ophs = sorted(df["operational_hours"].unique())
    return render_template("feed_data.html", sensors1=sensors1, sensors2=sensors2, sensors3=sensors3, ophs=ophs)

@app.route("/prediction")
def prediction():
    try:
        # Get parameters from the request
        sensor_1 = float(request.args.get("sensor_1", 0))
        sensor_2 = float(request.args.get("sensor_2", 0))
        sensor_3 = float(request.args.get("sensor_3", 0))
        operational_hours = float(request.args.get("operational_hours", 0))
        
        # Log the input values for debugging
        print(f"Received input values: sensor_1={sensor_1}, sensor_2={sensor_2}, sensor_3={sensor_3}, operational_hours={operational_hours}")

        # Create input array and scale it
        input_data = np.array([sensor_1, sensor_2, sensor_3, operational_hours]).reshape(1, -1)
        scaled_data = scaler.transform(input_data)

        # Log the scaled data for debugging
        print(f"Scaled Data: {scaled_data}")

        # Make prediction
        result = round(model.predict(scaled_data))
        
        # Log the prediction result
        result = model.predict(scaled_data)
        print(f"Prediction result: {result[0]}")

        # Render template with results
        return render_template("prediction.html", 
                               sensor_1=sensor_1, 
                               sensor_2=sensor_2, 
                               sensor_3=sensor_3, 
                               operational_hours=operational_hours, 
                               result=result)
    except Exception as e:
        # Log the exception and return an error message
        print(f"Error: {e}")
        return f"An error occurred: {e}", 500

if _name_ == "_main_":
    app.run(port=4996, debug=True)
