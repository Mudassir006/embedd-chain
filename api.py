import hashlib
import hmac
import os
import random
import string

from dotenv import load_dotenv
from embedchain import App as EmbedChainApp
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from supabase import create_client, Client

# Load environment variables from .env file
load_dotenv()

# Set environment variables
os.environ["TOGETHER_API_KEY"] = os.getenv("TOGETHER_API_KEY")
os.environ['ZILLIZ_CLOUD_URI'] = os.getenv('ZILLIZ_CLOUD_URI')
os.environ['ZILLIZ_CLOUD_TOKEN'] = os.getenv('ZILLIZ_CLOUD_TOKEN')
os.environ['PINECONE_API_KEY'] = os.getenv('PINECONE_API_KEY')
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Constants
HOST = '0.0.0.0'
PORT = 8000

# Initialize FastAPI app
app = FastAPI()

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Update with your frontend URL in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Dictionary to store EmbedChainApp instances by workspace ID
workspace_apps = {}


# Helper functions
def generate_workspace_id():
    """Generate a 12-character alphanumeric workspace ID."""
    return ''.join(random.choices(string.ascii_letters + string.digits, k=12))


def get_workspace_app(workspace_id: str):
    """Get or create an EmbedChainApp instance for the given workspace ID."""
    if workspace_id not in workspace_apps:
        workspace_apps[workspace_id] = EmbedChainApp.from_config(config_path="config.yaml")
    return workspace_apps[workspace_id]


async def verify_hmac_signature(request: Request, secret_key: str):
    """Verify HMAC signature to ensure the request is secure."""
    signature = request.headers.get("X-Signature")
    if not signature:
        raise HTTPException(status_code=403, detail="Signature missing")

    body = await request.body()
    computed_signature = hmac.new(secret_key.encode(), body, hashlib.sha256).hexdigest()

    if not hmac.compare_digest(computed_signature, signature):
        raise HTTPException(status_code=403, detail="Invalid signature")


# Pydantic models for request validation
class QuestionRequest(BaseModel):
    question: str
    session_id: str
    workspace_id: str
    supabase_url: str
    supabase_key: str


class TrainRequest(BaseModel):
    file_url: str
    data_type: str
    workspace_id: str
    supabase_url: str
    supabase_key: str


class QnARequest(BaseModel):
    question: str
    answer: str
    data_type: str
    workspace_id: str
    supabase_url: str
    supabase_key: str


class WorkspaceCreateRequest(BaseModel):
    workspace_name: str
    supabase_url: str
    supabase_key: str


# Endpoint to create a new workspace
@app.post("/api/workspace")
async def create_workspace(request: WorkspaceCreateRequest):
    workspace_id = generate_workspace_id()

    # Initialize Supabase client with provided credentials
    supabase: Client = create_client(request.supabase_url, request.supabase_key)

    # Save the workspace ID and name to Supabase
    supabase.table("workspaces").insert({"workspace_id": workspace_id, "workspace_name": request.workspace_name}).execute()

    return {"workspace_id": workspace_id, "workspace_name": request.workspace_name}


@app.post("/api/data")
async def get_data(request: QuestionRequest, req: Request):
    await verify_hmac_signature(req, os.getenv("HMAC_SECRET"))

    question = request.question
    workspace_id = request.workspace_id
    app_instance = get_workspace_app(workspace_id)

    try:
        answer = app_instance.search(question, num_documents=5, metadata={"workspace_id": workspace_id})
        if len(answer) == 0:  # empty answer check (hallucination)
            answer = app_instance.search(question, num_documents=5, metadata={"workspace_id": workspace_id})
        return answer
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train")
async def train(request: TrainRequest, req: Request):
    await verify_hmac_signature(req, os.getenv("HMAC_SECRET"))

    file_url = request.file_url
    data_type = request.data_type
    workspace_id = request.workspace_id
    app_instance = get_workspace_app(workspace_id)

    if not file_url or not data_type:
        raise HTTPException(status_code=400, detail="Both 'file_url' and 'data_type' are required")

    try:
        app_instance.add(file_url, data_type=data_type, metadata={"workspace_id": workspace_id})
        return {"status": "Training data added successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/train/qna")
async def train_qna(request: QnARequest, req: Request):
    await verify_hmac_signature(req, os.getenv("HMAC_SECRET"))

    question = request.question
    answer = request.answer
    data_type = request.data_type
    workspace_id = request.workspace_id
    app_instance = get_workspace_app(workspace_id)

    try:
        app_instance.add((question, answer), data_type=data_type, metadata={"workspace_id": workspace_id})
        return {"status": "Q&A pair added successfully"}
    except ValueError as ve:
        raise HTTPException(status_code=422, detail=str(ve))
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/api/data/list/{workspace_id}")
async def list_data(workspace_id: str, req: Request):
    await verify_hmac_signature(req, os.getenv("HMAC_SECRET"))

    app_instance = get_workspace_app(workspace_id)

    try:
        data_list = app_instance.list_data(metadata={"workspace_id": workspace_id})
        return {"data": data_list}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Endpoint to delete specific data from a workspace
@app.delete("/api/data/delete/{workspace_id}")
async def delete_data(workspace_id: str, file_name: str, supabase_url: str, supabase_key: str, req: Request):
    await verify_hmac_signature(req, os.getenv("HMAC_SECRET"))

    app_instance = get_workspace_app(workspace_id)

    try:
        app_instance.delete_data(file_name, metadata={"workspace_id": workspace_id})
        return {"status": f"Data {file_name} deleted successfully"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# Run the FastAPI app with Uvicorn
if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=HOST, port=int(os.getenv("PORT", default=5000)))
