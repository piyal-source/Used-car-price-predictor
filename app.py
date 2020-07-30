import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import make_column_transformer
from sklearn.pipeline import make_pipeline

df = pd.read_csv('quikr_data.csv')

app = Flask(__name__)

@app.route('/')
def index():
    company_list, fuel_type_list, model_dict = droplists()
    return render_template('index.html', company_list=company_list, fuel_type_list=fuel_type_list,
                           model_dict=model_dict)

def droplists():
    company_list = sorted(list(set(df['company'].values)))
    fuel_type_list = ['Petrol', 'Diesel', 'LPG']
    model_dict = dict()
    for i in company_list:
        model_dict[i] = []
        for j in sorted(list(set(df[df['company'] == i]['name'].values))):
            model_dict[i].append(j)
    return company_list,fuel_type_list,model_dict

@app.route('/predict', methods=['POST'])
def predict():
#    company_list, fuel_type_list, model_dict = droplists()
    company_name = request.form.get('company')
    model_name = request.form.get('model')
    year = request.form.get('year')
    kms_driven = request.form.get('kms')
    fuel_type = request.form.get('fuel')
    price = predictor(company_name,model_name,year,kms_driven,fuel_type)
    return render_template('index.html', price=price, company_name=company_name, fuel_type=fuel_type,
                           year=year, model_name=model_name, kms_driven=kms_driven)

def predictor(company_name, model_name, year, kms_driven, fuel_type):
    df1 = df.copy()
    X_test = pd.DataFrame(columns=['name', 'company', 'year', 'kms_driven', 'fuel_type'],
                          data=np.array([model_name, company_name, year, kms_driven, fuel_type]).reshape(1, 5))

    mean_model = df1.groupby('name').mean()['Price']
    df1['name'] = df1['name'].map(mean_model)
    X_test['name'] = mean_model[model_name]

    mean_company = df1.groupby('company').mean()['Price']
    df1['company'] = df1['company'].map(mean_company)
    X_test['company'] = mean_company[company_name]

    X = df1.iloc[:, [0, 1, 2, 4, 5]]
    y = df1['Price']

    reg = LinearRegression()
    ohe = OneHotEncoder(handle_unknown='ignore')
    column_trans = make_column_transformer((ohe, ['fuel_type']), remainder='passthrough')
    pipe = make_pipeline(column_trans, reg)

    pipe.fit(X, y)

#    prediction = car_data.predict(pd.DataFrame(columns=['name','company','year','kms_driven','fuel_type'],data=np.array([model_name,company_name,year,kms_driven,fuel_type]).reshape(1,5)))[0]
    prediction = pipe.predict(X_test)[0]
    del df1
    return str("{:,}".format(round(float(prediction),2)))


if __name__=="__main__":
    app.run(debug=True)
