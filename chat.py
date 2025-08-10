import random
import json
import torch
from model import ChatNet
from nltk_utils import bag_of_words, tokenize
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import pytz
from datetime import datetime

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load intents, model, and other necessary data
with open('intents.json', 'r') as f:
    intents = json.load(f)

FILE = "data.pth"
ERRORFILE = "errors.json"
data = torch.load(FILE)

input_size = data["input_size"]
hidden_size = data["hidden_size"]
output_size = data["output_size"]
all_words = data["all_words"]
tags = data["tags"]
model_state = data["model_state"]

model = ChatNet(input_size, hidden_size, output_size).to(device)
model.load_state_dict(model_state)
model.eval()

bot_name = "Rolland"

# Logs error queries
def error_messages(text):
    # Google Sheets API setup
    scope = ["https://spreadsheets.google.com/feeds", "https://www.googleapis.com/auth/drive"]
    creds = ServiceAccountCredentials.from_json_keyfile_name("static/chatbot-logger-458103-a6eab71d1676.json", scope)
    client = gspread.authorize(creds)

    # Open the spreadsheet
    sheet = client.open("ChatbotErrors").sheet1

    # Get current time
    local_timezone = pytz.timezone("America/New_York")
    timestamp = datetime.now(local_timezone).strftime("%Y-%m-%d %H:%M:%S")

    # Create the error message
    error_message = "I'm sorry, but I don't understand what you are trying to ask."

    # Append the error log to the sheet
    sheet.append_row([timestamp, text, error_message])


def get_response(text):
    # Tokenize the input and get the bag of words
    sentence = tokenize(text)
    x = bag_of_words(sentence, all_words)
    x = x.reshape(1, x.shape[0])
    x = torch.from_numpy(x)
                
    # Get model output
    output = model(x)
    _, predicted = torch.max(output, dim=1)
    tag = tags[predicted.item()]

    # Calculate probability
    probs = torch.softmax(output, dim=1)
    prob = probs[0][predicted.item()]
   
    # Return response based on the confidence
    if prob.item() > 0.75:
        for intent in intents["intents"]:
            if tag == intent["tag"]:
                return random.choice(intent["responses"])
                
    # Displays an error message and logs the query that generated
    # the message for investigation.
    else:
        error_messages(text)
        return f" I'm sorry, but I don't understand what you are trying to ask."
    


def chat():
    print("Let's chat! Type 'quit' to exit.")
    
    while True:
        # User input (the sentence to send to the bot)
        sentence = input("You: ")
        if sentence.lower() == "quit":
            print(f"{bot_name}: Goodbye!")
            break
        
        # Get bot response
        response = get_response(sentence)
        
        # Print the bot's response
        print(response)

# Start the chat
if __name__ == "__main__":
    chat()
