# Sentiment Analysis API
This repository contains code for a sentiment analysis API built using FastAPI and a pre-trained sentiment analysis model from Hugging Face Transformers.

## Overview
The sentiment analysis API analyzes the sentiment of input text and provides probabilities for negative, neutral, and positive sentiments.

## Installation
- install python version 3.11.9
- create virtual environment. cmd: "python -m venv env"
- Activate environment. windows cmd: "env\Scripts\activate"
- Install packages. "pip install fastapi uvicorn transformers torch scipy"

## Usage
- Run the API. "uvicorn predict:app --reload"
- Send a POST request to http://localhost:8000/predict with JSON payload containing the input text.Example request payload:
  {  
      "text": "The user acceptence test is verified and working as expected. Please proceed with deployment"
  }
- The API will return a JSON response with following payload:
  {
    "status": "Approved",
    "sentiment": "Positive",
    "confidence": 0.9916182160377502
  }
