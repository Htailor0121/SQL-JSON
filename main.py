from fastapi import FastAPI, Request
from pydantic import BaseModel
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from typing import Any, Optional
import os
import json
from sql_to_json_reporter import QueryConverter

app = FastAPI()

# TEMP: Allow CORS from all origins for debugging
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/ping")
async def ping():
    return {"message": "pong"}

class DBConfig(BaseModel):
    db_type: str
    host: str
    port: int
    user: str
    password: str
    database: str

class ChatRequest(BaseModel):
    query: str
    db_config: DBConfig

class ChatResponse(BaseModel):
    response: Any
    download_url: Optional[str] = None

class SchemaRequest(BaseModel):
    db_config: DBConfig

@app.post("/chat", response_model=ChatResponse)
async def chat_endpoint(request: ChatRequest):
    db_config = {
        'host': request.db_config.host,
        'port': request.db_config.port,
        'user': request.db_config.user,
        'password': request.db_config.password,
        'database': request.db_config.database
    }
    converter = QueryConverter(request.db_config.db_type, db_config)
    try:
        results = converter.process_query(request.query)
        if results is None:
            return ChatResponse(response="No results found.")
        # Find the latest result file generated
        files = [f for f in os.listdir('.') if f.endswith('.json') and '_results_' in f]
        if not files:
            return ChatResponse(response="No results file found.")
        latest_file = max(files, key=os.path.getctime)
        download_url = f"/download/{latest_file}"
        return ChatResponse(response="Query completed. Download your results:", download_url=download_url)
    except Exception as e:
        return ChatResponse(response=f"Error: {str(e)}")
    finally:
        converter.close()

@app.post("/schema")
async def schema_endpoint(request: SchemaRequest):
    try:
        db_config = {
            'host': request.db_config.host,
            'port': request.db_config.port,
            'user': request.db_config.user,
            'password': request.db_config.password,
            'database': request.db_config.database
        }
        converter = QueryConverter(request.db_config.db_type, db_config)
        schema = converter.get_schema()
        return schema
    except Exception as e:
        print("SCHEMA ERROR:", e)
        return {"error": str(e)}
    finally:
        try:
            converter.close()
        except Exception:
            pass

@app.get("/download/{filename}")
async def download_file(filename: str):
    file_path = os.path.join('.', filename)
    if not os.path.isfile(file_path):
        return {"error": "File not found."}
    return FileResponse(file_path, media_type='application/json', filename=filename) 