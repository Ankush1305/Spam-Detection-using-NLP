from flask import Flask,render_template,request
import pickle

# Load the Multinomial Naive Bayes model and CountVectorizer object from disk
filename = 'spam-sms-mnb-model.pkl'
classifier = pickle.load(open(filename, 'rb'))
cv = pickle.load(open('cv-transform.pkl','rb'))
app = Flask(__name__)

@app.route('/',methods=['GET','POST'])
def home():
     return render_template('index.html')

@app.route('/prediction',methods=['POST'])
def predict():
    if request.method=='POST':
        message=request.form['message']
        data=[message]
        vect=cv.transform(data).toarray()
        mypred=classifier.predict(vect)
        return render_template('result.html',result=mypred)
if __name__=='__main__':
    app.run(debug=True)




