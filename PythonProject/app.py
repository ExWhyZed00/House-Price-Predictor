import pickle
import pandas as pd
from flask import Flask, render_template, request
import numpy as np

app = Flask(__name__)
data= pd.read_csv('Cleaned_data.csv')
pipe = pickle.load(open('RidgeModel.pkl', 'rb'))

@app.route('/')
def index():
    locations = sorted(data['location'].unique())
    return render_template('index.html', location=locations)

@app.route('/predict', methods=['POST'])
def predict():
    locations = request.form.get('location')
    bhk = request.form.get('bhk')
    bath = request.form.get('bath')
    sqft = request.form.get('total_sqft')
    bhk = float(bhk)
    bath = float(bath)

    print(locations, bhk, bath, sqft)
    input = pd.DataFrame([[locations, bhk, bath, sqft]],columns=['location', 'bhk', 'bath', 'total_sqft'])
    prediction = pipe.predict(input)[0] * 1e5

    return str(np.round(prediction, 2))

if __name__ == '__main__':
    app.run(debug=True)

# import pickle
# import pandas as pd
# from flask import Flask, render_template, request
#
# app = Flask(__name__)
#
# # Load cleaned data and pretrained model
# data = pd.read_csv('Cleaned_data.csv')
# pipe = pickle.load(open('RidgeModel.pkl', 'rb'))
#
# @app.route('/')
# def index():
#     # Pass 'locations' to match the templates loop
#     locations = sorted(data['location'].unique())
#     print("Rendering template with locations:", locations)
#     return render_template('index.html', location=locations)
#
# @app.route('/predict', methods=['POST'])
# def predict():
#     # Retrieve and convert form inputs
#     location = request.form['location']
#     bhk = float(request.form['bhk'])
#     bath = float(request.form['bath'])
#     sqft = float(request.form['total_sqft'])
#
#     # Build DataFrame for the model
#     input_df = pd.DataFrame(
#         [[location, sqft, bath, bhk]],
#         columns=['location', 'total_sqft', 'bath', 'bhk']
#     )
#
#     # Make prediction and return as plain text
#     price = pipe.predict(input_df)[0]
#     return str(round(price, 2))
#
# if __name__ == '__main__':
#     app.run(debug=True)
