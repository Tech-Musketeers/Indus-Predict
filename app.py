from flask import Flask, request, render_template
import numpy as np
import pandas as pd
import pickle
from sklearn.preprocessing import StandardScaler
import os

app = Flask(__name__)

# Load the model and scaler
with open("equip_mnts.pkl", "rb") as f:
    model = pickle.load(f)

# Load the data for scaling
df = pd.read_csv(os.path.join(os.path.dirname(__file__), "templates", "machinery_data.csv"))
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
    return render_template("feed_data.html", sensors1 = sensors1, sensors2 = sensors2, sensors3 = sensors3,ophs=ophs)

@app.route("/prediction")
def prediction():
    sensor_1 = request.args.get("sensor_1")
    sensor_2 = request.args.get("sensor_2")
    sensor_3 = request.args.get("sensor_3")
    operational_hours = request.args.get("operational_hours")
    


    myinput = np.array([sensor_1, sensor_2, sensor_3, operational_hours]).reshape(1, 5)
    columns = ["sensor_1", "sensor_2", "sensor_3", "operational_hours"]
    mydata = pd.DataFrame(columns = columns, data = myinput)
    result = round(model.predict(mydata)[0,0], 2)

    return render_template("prediction.html", sensor_1 = sensor_1, sensor_2 = sensor_2, sensor_3 = sensor_3, operational_hours = operational_hours, result = result)




# @app.route('/')
# def index():
#     return render_template("index.html")

# @app.route('/reference_data')
# def reference_data():
#     # Display reference data from the CSV
#     reference_table = df.to_html(classes='table table-striped', index=False)
#     return render_template("reference_data.html", tables=[reference_table])

# @app.route('/feed_data')
# def feed_data():
#     if request.method == "POST":
#         try:
#             # Extract and convert form data
#             sensor_1 = float(request.form['sensor_1'])
#             sensor_2 = float(request.form['sensor_2'])
#             sensor_3 = float(request.form['sensor_3'])
#             operational_hours = float(request.form['operational_hours'])
            
#             # Scale and predict
#             feature = scaler.transform([[sensor_1, sensor_2, sensor_3, operational_hours]])
#             prediction = model.predict(feature)[0]
            
#             # Interpret prediction
#             prediction_text = "Yes" if prediction == 1 else "No"
            
#             return render_template("feed_data.html", Prediction_text=f"Maintenance Prediction is --->> {prediction_text}")
#         except (ValueError, KeyError) as e:
#             # Handle potential errors in form submission
#             return render_template("feed_data.html", error="Invalid input. Please enter valid numeric values for all sensors.")
#     else:
#         return render_template("feed_data.html")
    
# @app.route('/prediction')
# def prediction():
#     return render_template("prediction.html")


if __name__=="__main__":
    app.run(port=4996)