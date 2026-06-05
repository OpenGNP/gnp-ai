from fastapi import FastAPI
from pydantic import BaseModel
from typing import List

app = FastAPI()

class FeedbackRequest(BaseModel):
    feedback: List[str]

@app.get("/")
def root():
    return {"message": "AI service is running"}

@app.post("/analyze")
def analyze_feedback(data: FeedbackRequest):
    return {
        "total_feedback": len(data.feedback),
        "topics": [],
        "sentiment": [],
        "summary": "Analysis will be added here"
    }