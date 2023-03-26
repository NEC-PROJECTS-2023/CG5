from flask import Flask, request, render_template
import pickle
import pandas as pd
import numpy as np
app = Flask(__name__)
classifier=pickle.load(open('model.pkl','rb'))


@app.route("/")
def home():
    return render_template("front.html")

@app.route("/predict", methods = ['POST','GET'])
def predict():
    print(request.form)
    int_features=[int(x) for x in request.form.values()]
    final=[np.array(int_features)]
    print(int_features)
    prediction=classifier.predict_proba(final)
    output = '{0:.{1}f}'.format(prediction[0][1], 2)
    if output>str(0.5):
        return render_template('front.html',pred='Water is safe to drink'.format(output))
    else:
        return render_template('front.html',pred='water is unsafe to drink'.format(output))  

if __name__ == "__main__":
    
    app.run(debug=True)

