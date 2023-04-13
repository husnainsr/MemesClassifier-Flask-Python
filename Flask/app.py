#app.py
from flask import Flask, flash, request, redirect, url_for, render_template
import os
import urllib.request
from werkzeug.utils import secure_filename
import cv2
import pandas as pd
import pandas as pd
import string
import nltk.corpus
from textblob import TextBlob
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from nltk.stem import WordNetLemmatizer
from cleantext import clean
import numpy as np
from nltk.stem.snowball import SnowballStemmer
import re
from nltk.tokenize import word_tokenize
import os
import  pytesseract
from skimage import feature
from skimage import feature
from skimage.transform import resize
import numpy as np
from skimage.io import imread, imshow
import pickle
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report 
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier






app = Flask(__name__)

contractions = { 
"ain't": "am not",
"aren't": "are not",
"can't": "cannot",
"can't've": "cannot have",
"'cause": "because",
"could've": "could have",
"couldn't": "could not",
"couldn't've": "could not have",
"didn't": "did not",
"doesn't": "does not",
"don't": "do not",
"hadn't": "had not",
"hadn't've": "had not have",
"hasn't": "has not",
"haven't": "have not",
"he'd": "he would",
"he'd've": "he would have",
"he'll": "he will",
"he'll've": "he will have",
"he's": "he is",
"how'd": "how did",
"how'd'y": "how do you",
"how'll": "how will",
"how's": "how does",
"i'd": "i would",
"i'd've": "i would have",
"i'll": "i will",
"i'll've": "i will have",
"i'm": "i am",
"i've": "i have",
"isn't": "is not",
"it'd": "it would",
"it'd've": "it would have",
"it'll": "it will",
"it'll've": "it will have",
"it's": "it is",
"let's": "let us",
"ma'am": "madam",
"mayn't": "may not",
"might've": "might have",
"mightn't": "might not",
"mightn't've": "might not have",
"must've": "must have",
"mustn't": "must not",
"mustn't've": "must not have",
"needn't": "need not",
"needn't've": "need not have",
"o'clock": "of the clock",
"oughtn't": "ought not",
"oughtn't've": "ought not have",
"shan't": "shall not",
"sha'n't": "shall not",
"shan't've": "shall not have",
"she'd": "she would",
"she'd've": "she would have",
"she'll": "she will",
"she'll've": "she will have",
"she's": "she is",
"should've": "should have",
"shouldn't": "should not",
"shouldn't've": "should not have",
"so've": "so have",
"so's": "so is",
"that'd": "that would",
"that'd've": "that would have",
"that's": "that is",
"there'd": "there would",
"there'd've": "there would have",
"there's": "there is",
"they'd": "they would",
"they'd've": "they would have",
"they'll": "they will",
"they'll've": "they will have",
"they're": "they are",
"they've": "they have",
"to've": "to have",
"wasn't": "was not",
" u ": " you ",
" ur ": " your ",
" n ": " and "}

def cleaningText(text):
    stop = stopwords.words('english')
    text=text.lower()
    text=text.translate(str.maketrans('', '', string.punctuation)) #remove punctuation
    tokens = nltk.word_tokenize(text) #removing repated words
    ordered_tokens = set()
    result = []
    for word in tokens:
        if word not in ordered_tokens:
            ordered_tokens.add(word)
            result.append(word)
    text=" ".join(result)
    text= ''.join([i for i in text if not i.isdigit()]) #removing digits
    text= " ".join(text.split()) #removing extra spaces
    text = re.sub(r"(@\[A-Za-z0-9]+)|([^0-9A-Za-z \t])|(\w+:\/\/\S+)|^rt|http.+?", "", text) #remove extended asci etc
    gfg = TextBlob(text)
    text= gfg.correct()
    text = " ".join([word for word in text.split() if word not in (stop)]) #remove stopwords
    tokens = nltk.word_tokenize(text) #tokens
    lemmatizer = WordNetLemmatizer() #limitzer
    snowball = SnowballStemmer('english')#snowball
    t = [snowball.stem(t) for t in text]
    text="".join(t)
    text = text.strip()
    return text
pytesseract.pytesseract.tesseract_cmd=r"C:\Program Files\Tesseract-OCR\tesseract.exe"
def vectorizerr(txt):
    newDf=pd.DataFrame()
    df= pd.read_csv("./files/aug.csv")
    df = df.dropna(subset=['clean_text'])
    data=[txt]
    newDf["clean_text"]=data
    df1 = df.append(newDf, ignore_index = True)
    X=df1.clean_text
    vectorizer = TfidfVectorizer(lowercase=True,max_features=10167)
    X = vectorizer.fit_transform(X)
    df=pd.DataFrame(X.toarray(),columns=vectorizer.get_feature_names_out(),)
    array=df.iloc[-1]
    final=[array]
    return final

def textExtraction(path): 
    img=cv2.imread(path)
    img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    text=pytesseract.image_to_string(img)
    text=cleaningText(text)
    t=vectorizerr(text)
    return t
def imageConvertor(path):
    image1=imread(path,as_gray=True)
    image1=resize(image1,(150,150))
    image1 = feature.canny(image1)
    x=image1.astype('int32')
    image1= list(np.concatenate(x).flat)
    data=[image1]
    return data

    

    
def finalPrediction(txt_y,img_y):
    
    # 0 for neutral 1 for postive -1 for negative
    with open("./models/bagging_model_final.pkl",'rb') as file: 
        model1=pickle.load(file)
    with open("./models/decision_model_final.pkl",'rb') as file:
        model2=pickle.load(file)
    with open("./models/forest_model_final.pkl",'rb') as file:
        model3=pickle.load(file)
    with open("./models/gaussian_model_image.pkl",'rb') as file: 
        model4=pickle.load(file)
    with open("./models/gradient_model_image.pkl",'rb') as file:
        model5=pickle.load(file)
    with open("./models/logistic_model_image.pkl",'rb') as file:
        model6=pickle.load(file)
        
    print("----------------------lllll-----------------")
    zero=0
    ones=0
    neg=0
    finalAnswer=""
    pred1=model1.predict(txt_y)
    pred2=model2.predict(txt_y)
    pred3=model3.predict(txt_y)
    pred4=model4.predict(img_y)
    pred5=model5.predict(img_y)
    pred6=model6.predict(img_y)
    jelly=[pred1[0],pred2[0],pred3[0],pred4[0],pred5[0],pred6[0]]
    print(jelly)
    for i in jelly:
        if i==-1:
            neg+=1
        elif i==1:
            ones+=1
        elif i==0:
            zero+=1
    if(ones>zero and ones>neg):
        finalAnswer="Postive"
    elif(zero>ones and zero>neg):
        finalAnswer="Neutral"
    elif(neg>ones and neg>zero):
        finalAnswer="Negative"
    else:
        finalAnswer="Neutral"
        
    return finalAnswer
    # print(pred1[0],pred2[0],pred3[0],pred4[0],pred5[0],pred6[0])


@app.route('/', methods = ['GET'])
def upload():
    return render_template('index.html')

@app.route('/', methods = ['POST'])
def answer():
    img = request.files['img']
    imgpath = './images/' + img.filename
    img.save(imgpath)
    txt_y = textExtraction(imgpath) #text prediction
    img_y=imageConvertor(imgpath)   #image prediction
    prediction=finalPrediction(txt_y,img_y)
    # print(txt)
    p=finalPrediction(txt_y,img_y)
    return render_template('index.html',output=p)

if __name__ == "__main__":
    app.run(port=3000,debug=True)
    