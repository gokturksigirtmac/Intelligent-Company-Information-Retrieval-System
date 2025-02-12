# Intelligent Company Information Retrieval System

## Overview

The project is a FastAPI-based service designed to process company queries using OpenAI and LangGraph. This API allows users to submit queries and receive AI-generated responses.

## Features

- **Health Check Endpoint**: Verify if the API is running.
- **Query Processing**: Submit a query and receive AI-generated insights.
- **Follow-Up Queries**: Enhance query refinement by adding a topic for better responses.

## Installation

### Prerequisites

- Python 3.8+
- `pip` installed

### Clone the Repository

```sh
git clone https://github.com/gokturksigirtmac/Intelligent-Company-Information-Retrieval-System.git
cd llm-api
```

### Provide Credentials
Create .env file into root directory and use your OpenAI Api Key and Tavily Api Key 
```
OPENAI_API_KEY = "credential"
TAVILY_API_KEY = "credential"
```


## Running the API

Generate docker image
```
docker build -t XXX .
```

Docker run
```
docker run -p 8000:8000 --name XXX
```

Start the FastAPI server with:

```sh
app:app --host 0.0.0.0 --port 8000 
```

## API Endpoints

### Health Check

- **Endpoint**: `/health/`
- **Method**: `GET`
- **Response**:

```json
{
  "status": "ok"
}
```

### Process Query

- **Endpoint**: `/query/`
- **Method**: `POST`
- **Request Body**:

```json
{
  "query": "What is the company's latest update?"
}
```

- **Response**: AI-generated response based on the input query.

### Handle Follow-Up Queries

- **Endpoint**: `/query/follow-up/`
- **Method**: `POST`
- **Request Body**:

```json
{
  "query": "Give me more details about the latest update.",
  "topic": "Company finances"
}
```

- **Response**: AI-generated refined response incorporating the topic.
