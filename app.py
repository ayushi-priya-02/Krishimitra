from flask import Flask, render_template, request
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
import pandas as pd

app = Flask(__name__)

# Load dataset
data = pd.read_csv("Crop_recommendation.csv")

# Input and Output
X = data.drop("label", axis=1)
y = data["label"]

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestClassifier()
model.fit(X_train, y_train)


@app.route("/", methods=["GET", "POST"])
def home():

    prediction = ""
    irrigation = ""
    water = ""

    if request.method == "POST":

        N = float(request.form["N"])
        P = float(request.form["P"])
        K = float(request.form["K"])
        temperature = float(request.form["temperature"])
        humidity = float(request.form["humidity"])
        ph = float(request.form["ph"])
        rainfall = float(request.form["rainfall"])

        input_data = [[
            N,
            P,
            K,
            temperature,
            humidity,
            ph,
            rainfall
        ]]

        prediction = model.predict(input_data)[0]

        # Irrigation Logic
        if rainfall < 50:
            irrigation = "High"
            water = "8 mm/day"

        elif rainfall < 180:
            irrigation = "Moderate"
            water = "5 mm/day"

        else:
            irrigation = "Low"
            water = "2 mm/day"

    return render_template(
        "index.html",
        prediction=prediction,
        irrigation=irrigation,
        water=water
    )


if __name__ == "__main__":
    app.run(debug=True)