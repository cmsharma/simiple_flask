from flask import Flask
from flask import render_template, request
import joblib
import numpy as np

app = Flask(__name__)
app.config["TEMPLATES_AUTO_RELOAD"] = True

@app.route('/')
def index():
   #return "Welcome FLASk"
   return render_template('/index.html')

@app.route('/hello_page')
def myfunction():
   #return "Welcome FLASk"
   return render_template('/hello.html')

@app.route('/result',methods = ['POST','GET'])
def result():
    if request.method == 'POST':
        # By default form data comes in the form of a dictionary- key, value pairs
        form_data = request.form.to_dict()
        # {'sepal-length':"4.5","": }
        form_data_list = list(form_data.values())
        # Type cast data to float values
        form_data_list = list(map(float, form_data_list))
        # There are 4 features that's why we need to reshape it in a 1x4 array
        # You need to typecast list into a numpy array to typecast it.
        form_data_list = np.array(form_data_list).reshape(1,4)
        model = joblib.load('iris_model.sav')

        prediction = model.predict(form_data_list)
        return render_template("/result.html",flowerclass=prediction[0])

@app.route('/index')
def msg():
   return 'Hello World'

if(__name__=='__main__'):
    app.run(debug=True)







