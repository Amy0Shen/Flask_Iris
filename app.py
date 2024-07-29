import numpy as np
from flask import Flask, request, render_template
import pickle

app = Flask(__name__)
model = pickle.load(open('model.pkl', 'rb'))

@app.route('/')
def home():
    return render_template('index.html')


@app.route('/predict',methods=['POST'])
def predict():
    int_features = [float(x) for x in request.form.values()]
    final_features = [np.array(int_features)]
    prediction = model.predict(final_features)
    
    classes = ["Setosa", "Versicolor", "Virginica"]
    output = classes[prediction[0]]

    return render_template('index.html', prediction_text='Sepal should be from the class {}.'.format(output))

if __name__ == "__main__":
    app.run(debug=True)