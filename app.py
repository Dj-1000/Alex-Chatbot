from keras.models import load_model
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
import tensorflow as tf
import nltk
import pickle
import json
import numpy as np
import random
import datetime
import requests
import time
import billboard
import webbrowser
from pycricbuzz import Cricbuzz
from nltk.stem import WordNetLemmatizer
from google.oauth2 import service_account
from google.cloud import language_v1

lemmatizer = WordNetLemmatizer()

# Chat initialization
model = load_model("my_secondmodel.h5")
intents = json.loads(open("intents.json", encoding='utf-8').read())
words = pickle.load(open("words.pkl", "rb"))
classes = pickle.load(open("classes.pkl", "rb"))

app = Flask(__name__)
CORS(app)
app.static_folder = 'static'


def clean_up(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words


def create_bow(sentence, words):
    sentence_words = clean_up(sentence)
    bag = list(np.zeros(len(words)))

    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model):
    p = create_bow(sentence, words)
    res = model.predict(np.array([p]))[0]
    threshold = 0.8
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []

    for result in results:
        return_list.append({'intent': classes[result[0]], 'prob': str(result[1])})
    return return_list


def get_response(return_list, intents_json):
    if len(return_list) == 0:
        tag = 'noanswer'
    else:
        tag = return_list[0]['intent']
    if tag == 'datetime':
        return [time.strftime("%A"), time.strftime("%d %B %Y"), time.strftime("%H:%M:%S")]

    if tag == 'weather':
        api_key = '987f44e8c16780be8c85e25a409ed07b'
        base_url = "http://api.openweathermap.org/data/2.5/weather?"
        city_name = input("Enter city name: ")
        complete_url = base_url + "appid=" + api_key + "&q=" + city_name
        response = requests.get(complete_url)
        x = response.json()
        return [round(x['main']['temp'] - 273, 2), round(x['main']['feels_like'] - 273, 2), x['weather'][0]['main']]

    if tag == 'song':
        chart = billboard.ChartData('hot-100')
        songs = []
        for i in range(10):
            song = chart[i]
            songs.append({'title': song.title, 'artist': song.artist})
        return songs

    if tag == 'cricket':
        c = Cricbuzz()
        matches = c.matches()
        cricket_matches = []
        for match in matches:
            cricket_matches.append({'srs': match['srs'], 'mnum': match['mnum'], 'status': match['status']})
        return cricket_matches

    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if tag == i['tag']:
            result = random.choice(i['responses'])
            return result

    return "I'm sorry, I don't have a response for that."


def chat_response(text):
    msg = list()
    msg.append(str(text))
    return_list = predict_class(text,model)
    response = get_response(return_list,intents)
    return response


def analyze_sentiment(text):
    SERVICE_ACCOUNT_FILE = r"D:\imp\glc nlp sevice account key.json"
    credentials = service_account.Credentials.from_service_account_file(SERVICE_ACCOUNT_FILE)
    client = language_v1.LanguageServiceClient(credentials=credentials)
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



@app.route("/")
def home():
    return render_template("index.html")


@app.route("/get", methods=["POST"])
def response():
    msg = list()
    user_input = str(request.json['text'])
    msg.append(user_input)
    ans = song_emotion(user_input)
    response = chat_response(user_input)
    return jsonify({'response': response, 'songs': ans})


if __name__ == "__main__":
    app.run()
