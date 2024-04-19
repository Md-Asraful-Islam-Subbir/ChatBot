import os
import json
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS
from chat import get_response
from train import update_intents_file, train_model

current_directory = os.getcwd()
print(f"Current working directory: {current_directory}")

os.chdir(os.path.dirname(os.path.abspath(__file__)))




app = Flask(__name__)
CORS(app)

@app.route("/")
def index_get():
    return render_template("base.html")

@app.route("/predict", methods=["POST"])
def predict():
    text = request.json.get('user_input')


    response = get_response(text)
    message = {"answer": response}
    return jsonify(message)

@app.route("/train_new", methods=["POST"])
def train_new():
    question = request.json.get('user_input')
    answer= request.json.get('open_ai_res')
    
    new_data = {
        'tag': [question], 
        'patterns': [question], 
        'responses':[answer]
    }
    update_intents_file(new_data)
    train_model()  
    
    return jsonify({"message": "Training completed successfully."})

if __name__ == "__main__":
    app.run(port=5000, debug=True)
    