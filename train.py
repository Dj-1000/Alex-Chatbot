import keras
import keras
import tensorflow as tf
import nltk
import pickle
import json
import numpy as np
from keras.models import Sequential
from keras.layers import Dense,Dropout,Activation
import random
import datetime
import requests
import time
import billboard
# from pygame import mixer
# from googlesearch import *
import webbrowser
from pycricbuzz import Cricbuzz

nltk.download('punkt')
nltk.download('wordnet')

from nltk.stem import WordNetLemmatizer
lemmatizer=WordNetLemmatizer()

msg = list()
words=[]
classes=[]
documents=[]
ignore=['?','!',',',"'s"]

data_file=open('intents2.json',encoding='utf-8').read()
intents=json.loads(data_file)

for intent in intents['intents']:
    for pattern in intent['patterns']:
        w=nltk.word_tokenize(pattern)
        words.extend(w)
        documents.append((w,intent['tag']))
        
        if intent['tag'] not in classes:
            classes.append(intent['tag'])

words=[lemmatizer.lemmatize(word.lower()) for word in words if word not in ignore]
words=sorted(list(set(words)))
classes=sorted(list(set(classes)))
pickle.dump(words,open('words.pkl','wb'))
pickle.dump(classes,open('classes.pkl','wb'))

#training data
training=[]
output_empty=[0]*len(classes)

for doc in documents:
    bag=[]
    pattern=doc[0]
    pattern=[ lemmatizer.lemmatize(word.lower()) for word in pattern ]
    
    for word in words:
        if word in pattern:
            bag.append(1)
        else:
            bag.append(0)
    output_row=list(output_empty)
    output_row[classes.index(doc[1])]=1
    
    training.append([bag,output_row])
    
random.shuffle(training)
training=np.array(training)  
X_train=list(training[:,0])
y_train=list(training[:,1]) 

#Model
model=Sequential()
model.add(Dense(128,activation='relu',input_shape=(len(X_train[0]),)))
model.add(Dropout(0.5))
model.add(Dense(64,activation='relu'))
model.add(Dense(64,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(len(y_train[0]),activation='softmax'))

adam=keras.optimizers.Adam(0.001)
model.compile(optimizer=adam,loss='categorical_crossentropy',metrics=['accuracy'])
weights=model.fit(np.array(X_train),np.array(y_train),epochs=300,batch_size=10,verbose=1)    
model.save('my_secondmodel.h5',weights)

from keras.models import load_model
model = load_model('my_secondmodel.h5')
intents = json.loads(open('intents2.json',encoding='utf-8').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

#Predict
def clean_up(sentence):
    sentence_words=nltk.word_tokenize(sentence)
    sentence_words=[ lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

def create_bow(sentence,words):
    sentence_words=clean_up(sentence)
    bag=list(np.zeros(len(words)))
    
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                bag[i] = 1
    return np.array(bag)

def predict_class(sentence,model):
    p=create_bow(sentence,words)
    res=model.predict(np.array([p]))[0]
    threshold=0.8
    results=[[i,r] for i,r in enumerate(res) if r>threshold]
    results.sort(key=lambda x: x[1],reverse=True)
    return_list=[]
    
    for result in results:
        return_list.append({'intent':classes[result[0]],'prob':str(result[1])})
    return return_list


def get_response(return_list,intents_json):
    
    if len(return_list)==0:
        tag = 'noanswer'
    else:    
        tag=return_list[0]['intent']
    if tag=='datetime':        
        print(time.strftime("%A"))
        print (time.strftime("%d %B %Y"))
        print (time.strftime("%H:%M:%S"))

        
    if tag=='weather':
        api_key='987f44e8c16780be8c85e25a409ed07b'
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        city_name = input("Enter city name : ")
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url) 
        x=response.json()
        print('Present temp.: ',round(x['main']['temp']-273,2),'celcius ')
        print('Feels Like:: ',round(x['main']['feels_like']-273,2),'celcius ')
        print(x['weather'][0]['main'])

        
    if tag=='song':
        chart = billboard.ChartData('hot-100')
        print('The top 10 songs at the moment are:')
        for i in range(10):
            song=chart[i]
            print(song.title,'- ',song.artist)
            
    if tag=='cricket':
        c = Cricbuzz()
        matches = c.matches()
        for match in matches:
            print(match['srs'],' ',match['mnum'],' ',match['status'])
    

    list_of_intents = intents_json['intents']    
    for i in list_of_intents:
        if tag==i['tag'] :
            result= random.choice(i['responses'])
    return result





from google.oauth2 import service_account
from google.cloud import language_v1
# Path to your Google Cloud service account key JSON file
SERVICE_ACCOUNT_FILE = r"{google cloud nlp api service account file}"

# Initialize the client and authenticate using service account credentials
credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
client = language_v1.LanguageServiceClient(credentials=credentials)

#For calculating sentiment score
def analyze_sentiment(text):
    # Create a Document object and specify the text and language
    # document = language_v1.Document(content=text, type_=language_v1.Document.Type_.PLAIN_TEXT)
    document = language_v1.Document(content=text, type_=language_v1.Document.Type.PLAIN_TEXT)

    # Specify the encoding type for sentiment analysis
    encoding_type = language_v1.EncodingType.UTF8

    # Analyze sentiment
    response = client.analyze_sentiment(request={'document': document, 'encoding_type': encoding_type})

    # Get sentiment score and magnitude
    sentiment_score = response.document_sentiment.score
    sentiment_magnitude = response.document_sentiment.magnitude

    return sentiment_score, sentiment_magnitude


def detect_emotion(text):
    sentiment_scores, sentiment_magnitude = analyze_sentiment(text)
    # Determine the dominant emotion based on the sentiment scores
    if sentiment_scores >= 0.05:
        emotion = 'Happy'
    elif sentiment_scores <= -0.05:
        emotion = 'Sad'
    elif sentiment_scores >= 0.2:
        emotion = 'Love'
    elif sentiment_scores <= -0.2:
        emotion = 'Anger'
    elif sentiment_scores<= -0.1:
        emotion = 'Fear'
    else:
        emotion = 'Neutral'

    return emotion

def song_emotion(text):
    len1 = len(text)
    dic1 = dict()
    emotion = detect_emotion(text)
    dic1['emotion'] = emotion

    import requests
    url=f"http://ws.audioscrobbler.com/2.0/?method=tag.gettoptracks&tag={emotion}&api_key=039a0f5d4baf90c8df5eebfd16bfe4c2&format=json&limit=10"
    response = requests.get(url)
    payload = response.json()
    for i in range(10):
        r=payload['tracks']['track'][i]
        dic1[r['name']] = r['url']
    return dic1



def chat_response(text):
    msg = list()
    msg.append(text)
    return_list=predict_class(text,model)
    response = get_response(return_list,intents)
    return response





while(1):
    x=input("User: ")
    res = chat_response(x)
    print("Alex : "+res)
    ans = song_emotion(x)
    print("Emotion : "+ans['emotion'])
    if x.lower() in ['bye','goodbye','get lost','see you','go','exit','hush now']:  
        break

# def song_retrieve(text):
#     ans = song_emotion(text)
#     return ans

    

ans = song_emotion(str(msg))
print("Emotion : "+ans['emotion'])
ans.pop('emotion')
lst = list(ans.keys())
print("Song Recommendations : ")
for i in range(10):
    print("Song_name : "+lst[i])
    print("Song_URL : "+ans[lst[i]])




