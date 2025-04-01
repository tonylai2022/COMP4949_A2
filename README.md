# 🧠 Local GGUF LLM API with FastAPI

This project runs a local Large Language Model (LLM) using a `.gguf` file (e.g., TinyLlama or Zephyr or your fine-tuned model) with [`llama-cpp-python`](https://github.com/abetlen/llama-cpp-python) and exposes an API via [FastAPI](https://fastapi.tiangolo.com/).

It's designed to integrate with a React frontend or any app that wants to generate natural language responses locally — **no cloud APIs required**!

---

## 🚀 Features

- Loads quantized `.gguf` models 
- FastAPI backend with `/generate` POST route
- CORS enabled for `http://localhost:3000` (React support)
- Easy setup, portable and lightweight

---

## 🛠 Setup Instructions

### 1. Create a Virtual Environment

```bash
python -m venv venv
venv\Scripts\activate      # Windows
# OR
source venv/bin/activate   # Mac/Linux

pip install fastapi uvicorn llama-cpp-python
uvicorn main:app --host 0.0.0.0 --port 8000
```

---

## 🧪 API Usage

### POST `/generate`

**Request Body:**

```json
{
  "prompt": "Suggest 3 outdoor activities for sunny weather in Vancouver."
}
```

**Response:**

```json
{
  "response": "• Hike at Grouse Mountain\n• Kayak in Deep Cove\n• Bike the Seawall"
}
```

---

## ⚛️ React Integration

Example using `axios`:

```javascript
const response = await axios.post("http://localhost:8000/generate", {
  prompt: "What to do in Whistler on a snowy day?"
});
console.log(response.data.response);
```

---

## ✅ Notes

- Works offline — all inference is done locally.
- Make sure `model.gguf` is present in the same folder.
- Supports models from TheBloke on Hugging Face.

---

## 💡 Recommended Small Models

- **TinyLlama 1.1B Chat (Q4_K_M)**
- **Zephyr 7B Alpha GGUF**
