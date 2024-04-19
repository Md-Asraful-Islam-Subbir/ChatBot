import random
import json

import torch

from model import NeuralNet
from nltk_utils import bag_of_words, tokenize

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

with open('intents.json', 'r') as json_data:
    intents = json.load(json_data)

FILE = "data.pth"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data['all_words']
tags = data['tags']
model_state = data["model_state"]

model = NeuralNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Sam"

def get_response(msg):
    sentence = tokenize(msg)
    X = bag_of_words(sentence, all_words)
    X = X.reshape(1, X.shape[0])
    X = torch.from_numpy(X).to(device)

    output = model(X)
    _, predicted = torch.max(output, dim=1)

    tag = tags[predicted.item()]
    
    print(f"Predicted Tag: {tag}")

    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
    print(f"Confidence: {prob.item()}")
    if prob.item() > 0.95:
        for intent in intents['intents']:
            intent_tag = intent["tag"]
            if isinstance(intent_tag, list):
                # Check each element of the list
                for element in intent_tag:
                    if tag.lower().strip() == element.lower().strip():
                        return random.choice(intent['responses'])
            else:
                # Handle the case where the intent tag is not a list
                if tag.lower().strip() == intent_tag.lower().strip():
                    return random.choice(intent['responses'])
    return "-99999"


if __name__ == "__main__":
    print("Let's chat! (type 'quit' to exit)")
    while True:
        # sentence = "do you use credit cards?"
        sentence = input("You: ")
        if sentence == "quit":
            break

        resp = get_response(sentence)
        print(resp)

