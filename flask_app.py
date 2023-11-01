from flask import Flask, request, jsonify
import fasttext
import re
import nltk
from nltk.tokenize.toktok import ToktokTokenizer
from nltk.corpus import stopwords
import unicodedata
from emoji import demojize
import json
from bp import better_profanity
from sentence_transformers import SentenceTransformer, util


app = Flask(__name__)

# methods
def text_cleaning(text_data, stop_words, lemmatizer):
    text_data = unicodedata.normalize('NFKD', text_data).encode('ascii', 'ignore').decode('utf-8', 'ignore')
    text_data = text_data.lower()
    text_data = demojize(text_data)
    pattern_punct = re.compile(r'([.,/#!$%^&*?;:{}=_`~()+-])\1{1,}')
    text_data = pattern_punct.sub(r'\1', text_data)
    text_data = re.sub(' {2,}',' ', text_data)
    text_data = re.sub(r"[^a-zA-Z?!]+", ' ', text_data)
    text_data = str(text_data)
    tokenizer = ToktokTokenizer()
    text_data = tokenizer.tokenize(text_data)
    text_data = [item for item in text_data if item not in stop_words]
    text_data = [lemmatizer.lemmatize(word = w, pos = 'v') for w in text_data]
    text_data = ' '.join (text_data)
    return text_data

def load_shit():
    # nltk.download('omw-1.4')
    nltk.download('stopwords')
    stop_words = set(stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    model = fasttext.load_model('/home/ayekaunic/mysite/profanity_model_eng.bin')
    nltk.download('wordnet')

    return stop_words, lemmatizer, model

def cosine_similarity(sentence1, sentence2, model):

    embedding_1 = model.encode(sentence1, convert_to_tensor = True)

    similarities = []
    embedding_2 = model.encode(sentence2, convert_to_tensor=True)

    similarity = util.pytorch_cos_sim(embedding_1, embedding_2)
    similarities.append(similarity.item())

    return similarities

def load_stuff():
    stop_words = set(stopwords.words('english'))
    lemmatizer = nltk.stem.WordNetLemmatizer()
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    nltk.download('wordnet')
    return stop_words, lemmatizer, model



# greetings
@app.route('/', methods = ["GET"])
def greetings():
    data_set = {'greetings': 'hello, world! :D'}
    json_dump = json.dumps(data_set)
    return json_dump


# english profanity
@app.route('/profanity/english', methods=['POST', 'GET'])
def check_profanity():
    try:
        data = request.get_json()
        user_input = data.get('input', '')
        # user_input = str(request.args.get('input'))
        stop_words, lemmatizer, model = load_shit()
        user_input = text_cleaning(user_input, stop_words, lemmatizer)
        labels, probabilities = model.predict(user_input, k=2)
        result = []
        result.append(user_input)
        for label, probability in zip(labels, probabilities):
            if label[9:] == "1":
                result.append({"label": "Profane", "probability": round(probability * 100, 1)})
            else:
                result.append({"label": "Clean", "probability": round(probability * 100, 1)})

        return jsonify({"results": result})
    except Exception as e:
        return jsonify({"error": str(e)})


# roman urdu
@app.route('/profanity/romanUrdu', methods=['POST', 'GET'])
def profanity_check():
    try:
        data = request.get_json()
        user_input = data.get('input', '')
        # user_input = str(request.args.get('input'))
        profanity = better_profanity.Profanity()
        return jsonify({"Profane": profanity.contains_profanity(user_input)})
    except Exception as e:
        return jsonify({"error": str(e)})

# similarty
@app.route('/similarity', methods=['POST'])
def check_duplicity():
    try:
        data = request.get_json()
        title1 = data.get("title1", "")
        description1 = data.get("description1", "")
        foi1 = data.get("foi1", "")

        title2 = data.get("title2", "")
        description2 = data.get("description2", "")
        foi2 = data.get("foi2", "")

        stop_words, lemmatizer, model = load_stuff()

        score = cosine_similarity(title1, title2, model)
        total_score = [x * 1.0 for x in score]

        clean_description1 = text_cleaning(description1, stop_words, lemmatizer)
        clean_description2 = text_cleaning(description2, stop_words, lemmatizer)
        score = cosine_similarity(clean_description1, clean_description2, model)
        total_score[0] += score[0] * 1.0

        score = cosine_similarity(foi1, foi2, model)
        total_score[0] += score[0] * 1.0

        average_score = total_score[0] / 3
        average_score = average_score * 0.369 + 0.631

        response = {
        "similarity": average_score
        }
        return jsonify(response)

    except Exception as e:
        return jsonify({"error": str(e)})