from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from llama_cpp import Llama
import os

# ✅ Set model path
MODEL_PATH = "model.gguf"  

# ✅ Make sure the model exists
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model not found at: {MODEL_PATH}")

# ✅ Load the LLM
llm = Llama(
    model_path=MODEL_PATH,
    n_ctx=512,
    n_threads=8,
    use_mlock=False
)

# ✅ Initialize FastAPI app
app = FastAPI()

# ✅ Enable CORS for your React frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],  # React dev server
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Optional: root endpoint for test
@app.get("/")
def home():
    return {"message": "GGUF API is running!"}

# ✅ POST /generate endpoint
@app.post("/generate")
async def generate(request: Request):
    data = await request.json()
    prompt = data.get("prompt", "")
    output = llm(prompt=prompt, max_tokens=128)
    return {"response": output["choices"][0]["text"]}
