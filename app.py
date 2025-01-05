from flask import Flask, render_template, request
import pickle
import numpy as np

model = pickle.load(open('my_iris_model.pkl','rb'))
app = Flask(__name__)
@app.route('/')
def man():
    return render_template('home.html')
@app.route('/Predict',methods=['POST'])
def home():
    data1 = float(request.form['a'])
    data2 = float(request.form['b'])
    data3 = float(request.form['c'])
    data4 = float(request.form['d'])
    ar = np.array([[data1,data2,data3,data4]])
    pred = model.predict(ar)
    return render_template('prediction.html',data=pred)
if __name__ == "__main__":
    app.run(debug=True)