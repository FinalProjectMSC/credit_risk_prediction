from flask import Flask, request, render_template, jsonify
from prediction import PredictionPipeline

app = Flask(__name__)

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict",methods= ['POST',"GET"])
def predict():
    if request.method == "POST":
        person_age = request.form.get("person_age")
        person_income = request.form.get("person_income")
        person_home_ownership = request.form.get("person_home_ownership")
        person_emp_length = request.form.get("person_emp_length")
        loan_intent = request.form.get("loan_intent")
        loan_grade = request.form.get("loan_grade")
        loan_amnt = request.form.get("loan_amnt")
        loan_int_rate = request.form.get("loan_int_rate")
        loan_percent_income = request.form.get("loan_percent_income")
        cb_person_default_on_file = request.form.get("cb_person_default_on_file")
        cb_person_cred_hist_length = request.form.get("cb_person_cred_hist_length")

        pipeline = PredictionPipeline(person_age, person_income, person_home_ownership, person_emp_length, loan_intent, loan_grade, loan_amnt, loan_int_rate, loan_percent_income, cb_person_default_on_file, cb_person_cred_hist_length)

        y_pred = pipeline.predict()
        y_pred = round(y_pred[0],2)
        print(y_pred)
        return render_template("index.html",loan_status = y_pred)

if __name__ == "__main__":
    app.run(debug=True)