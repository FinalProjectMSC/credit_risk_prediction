import pickle
import pandas as pd

class PredictionPipeline:
    def __init__(self,person_age:int,person_income:int,person_home_ownership: str,person_emp_length: int,loan_intent: str,loan_grade: str,loan_amnt: int,loan_int_rate: float,loan_percent_income:float,cb_person_default_on_file: str,cb_person_cred_hist_length:int):
        self.person_age = person_age
        self.person_income = person_income
        self.person_home_ownership = person_home_ownership
        self.person_emp_length = person_emp_length
        self.loan_intent = loan_intent
        self.loan_grade = loan_grade
        self.loan_amnt = loan_amnt
        self.loan_int_rate = loan_int_rate
        self.loan_percent_income = loan_percent_income
        self.cb_person_default_on_file = cb_person_default_on_file
        self.cb_person_cred_hist_length = cb_person_cred_hist_length

        input_df = {
            "person_age":[self.person_age],
            "person_income":[self.person_income],
            "person_home_ownership":[self.person_home_ownership],
            "person_emp_length":[self.person_emp_length],
            "loan_intent":[self.loan_intent],
            "loan_grade":[self.loan_grade],
            "loan_amnt":[self.loan_amnt],
            "loan_int_rate":[self.loan_int_rate],
            "loan_percent_income":[self.loan_percent_income],
            "cb_person_default_on_file":[self.cb_person_default_on_file],
            "cb_person_cred_hist_length":[self.cb_person_cred_hist_length]
        }

        self.final_df = pd.DataFrame(input_df)
        
    def predict(self):  
        
        with open("models/credict_risk_preproessor.pkl","rb") as pickle_file:
            self.preprocessor = pickle.load(pickle_file)

        with open("models/credit_risk_model.pkl","rb") as pickle_file:
            self.model = pickle.load(pickle_file)

        final_inputs = self.preprocessor.transform(self.final_df)
        y_pred = self.model.predict(final_inputs)
        print(y_pred)
        return y_pred


if __name__ == "__main__":
    prediction = PredictionPipeline(22,59000,"RENT",123,"PERSONAL",'D',35000,16.02, 0.59,"Y",3)
    prediction.predict()