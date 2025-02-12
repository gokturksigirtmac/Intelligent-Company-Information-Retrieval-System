from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import uvicorn
from app.agent import process_user_input
import sys
import os

app = FastAPI(title="LLM API", description="API for processing company queries using OpenAI and LangGraph.", version="1.0")

# FastAPI Request Model
class QueryRequest(BaseModel):
    query: str

class FollowUpRequest(BaseModel):
    query: str
    topic: str
    
# FastAPI Endpoints
@app.get("/health/", summary="Health Check")
async def health_check():
    return {"status": "ok"}

@app.post("/query/", summary="Process User Query")
async def process_query(request: QueryRequest):
    response = process_user_input(request.query)

    return response

@app.post("/query/follow-up/", summary="Handle Follow-Up Queries")
async def process_followup(request: FollowUpRequest):
    refined_query = f"{request.query} {request.topic}"  
    response = process_user_input(refined_query)
    
    return response

# Run the API
if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)