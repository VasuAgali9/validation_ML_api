# Bring in the neccessary packages
from fastapi import FastAPI, HTTPException, UploadFile, File
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch
import re
from email.parser import BytesParser

app = FastAPI()

# Load the sentiment analysis model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# request accepted model
# class SentimentRequest(BaseModel):
#     text: str

# response model
class SentimentResponse(BaseModel):
    status: str
    sentiment: str
    confidence: float

# # Post handler
# @app.post("/predict", response_model=SentimentResponse)
# async def analyze_sentiment(request: SentimentRequest):
#     prediction = predict(request.text)
#     return prediction

@app.post("/validate")
async def decode_attachement(attachement: UploadFile = File(...)):
    uploaded_email = attachement
    contents = await uploaded_email.read()
    email_data = parse_email(contents)
    return email_data

def parse_email(content: bytes):
    email_message = BytesParser().parsebytes(content)
    subject = email_message.get('Subject')
    sender = email_message.get('From')
    recipient = email_message.get('To')
    date = email_message.get('Date')
    body = ""

    if email_message.is_multipart():
        for part in email_message.walk():
            content_type = part.get_content_type()
            if content_type == 'text/plain' or content_type == 'text/html':
                body += part.get_payload(decode=True).decode(part.get_content_charset(), 'ignore')
    
            # elif content_type.startswith('image/'):
            #     # Handle images in the body
            #     image_data = part.get_payload(decode=True)
            #     image_name = part.get_filename() or 'image.jpg'
            #     with open(image_name, 'wb') as image_file:
            #         image_file.write(image_data)

    # else:
    #     content_type = email_message.get_content_type()
    #     if content_type == 'text/plain' or content_type == 'text/html':
    #         body = email_message.get_payload(decode=True).decode(email_message.get_content_charset(), 'ignore')
    body= body.split("On")
    body = re.sub(r'[\n\t\r]', '', body[0])
    image = "[image: image.png]" in body
    pridiction = predict(body)
    return {
        "pridiction" : pridiction,
        "subject" : subject,
        "sender" : sender,
        "recipient": recipient,
        "date": date,
        "image": image,
        "Body": body
    }

def predict(text: str):
    encoded_text = tokenizer(text, return_tensors='pt')
    # Perform sentiment analysis
    output = model(**encoded_text)
    scores = output.logits[0].detach().numpy()
    scores = softmax(scores)
    print(f"scores:  {scores}")
    sentiment_labels = ["negative", "neutral", "positive"]
    max_score_index = scores.argmax().item()
    sentiment_label = sentiment_labels[max_score_index]
    status = "Rejected" if max_score_index == 0 else "Approved"
    response = SentimentResponse(
        status=status,
        sentiment=sentiment_label,
        confidence=float(scores[max_score_index])
            )
    return response