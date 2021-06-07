from flask import Flask,render_template,url_for,request
from sklearn.naive_bayes import MultinomialNB
import joblib 
import pickle
from model import text_process

app = Flask(__name__)


@app.route('/')
def home():
	return render_template('home.html')

@app.route('/predict',methods=['POST'])
def predict():
    
    prediction_model = joblib.load('saved_model/SMS_spam_model.pkl')
    
    if request.method == 'POST':
        message = request.form['message']
        data = [message]
        
        my_prediction = prediction_model.predict(data)
    return render_template('result.html', prediction = my_prediction)



if __name__ == '__main__':
    app.run(port=5000)