# Bring in the neccessary packages
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import torch

app = FastAPI()

# Load the sentiment analysis model
MODEL = "cardiffnlp/twitter-roberta-base-sentiment"
tokenizer = AutoTokenizer.from_pretrained(MODEL)
model = AutoModelForSequenceClassification.from_pretrained(MODEL)

# request accepted model
class SentimentRequest(BaseModel):
    text: str

# response model
class SentimentResponse(BaseModel):
    status: str
    sentiment: str
    confidence: float

# Post handler
@app.post("/predict", response_model=SentimentResponse)
async def analyze_sentiment(request: SentimentRequest):
    try:
        # Tokenize the input text
        encoded_text = tokenizer(request.text, return_tensors='pt')
        # Perform sentiment analysis
        output = model(**encoded_text)
        scores = output.logits[0].detach().numpy()
        scores = softmax(scores)
        print(scores)
        # Prepare response
        if (scores[0] > (scores[1] and scores[2])) :
            response = SentimentResponse(
                status = "Rejected", 
                sentiment = "negative", 
                confidence = float(scores[0])
                )
        elif(scores[1] > (scores[0] and scores[2])):
            response = SentimentResponse(
                status = "Approved", 
                sentiment = "neutral", 
                confidence = float(scores[1])
                )
        else:
            response = SentimentResponse(
                status = "Approved", 
                sentiment = "Positive", 
                confidence = float(scores[2])
                )
        
        return response
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))