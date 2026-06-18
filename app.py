import os
import torch
from fastapi import FastAPI, Request, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from llama_cpp import Llama
from huggingface_hub import hf_hub_download
from dotenv import load_dotenv
import uvicorn
import asyncio

load_dotenv()

app = FastAPI(title="Qwen2.5-7B Chat")

# templates 설정
templates = Jinja2Templates(directory="templates")

print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# ==================== 모델 로드 ====================
print("Qwen2.5-7B 모델 로드 중... (최초 실행 시 다운로드됩니다)")
hf_hub_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q5_0-00001-of-00002.gguf"
)
hf_hub_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q5_0-00002-of-00002.gguf"
)

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q5_0-00001-of-00002.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
    verbose=False
)
print("✅ 모델 로드 완료!")

# ==================== WebSocket ====================
class ConnectionManager:
    def __init__(self):
        self.active_connections: list[WebSocket] = []

    async def connect(self, websocket: WebSocket):
        await websocket.accept()
        self.active_connections.append(websocket)

    def disconnect(self, websocket: WebSocket):
        if websocket in self.active_connections:
            self.active_connections.remove(websocket)

manager = ConnectionManager()


@app.get("/", response_class=HTMLResponse)
async def read_root(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})


@app.websocket("/ws/chat")
async def websocket_endpoint(websocket: WebSocket):
    await manager.connect(websocket)
    try:
        while True:
            data = await websocket.receive_json()
            user_message = data.get("message", "")

            if not user_message:
                continue

            messages = [
                {"role": "system", "content": "당신은 친절하고 똑똑한 친구같은 AI 개인 비서입니다. 한국어로 자연스럽게 답변해주세요."}
            ]
            if "history" in data:
                messages.extend(data["history"])

            messages.append({"role": "user", "content": user_message})

            response = llm.create_chat_completion(
                messages=messages,
                temperature=0.7,
                top_p=0.9,
                max_tokens=1024,
                repeat_penalty=1.05,
                stream=True
            )

            full_response = ""
            for chunk in response:
                if "choices" in chunk and chunk["choices"]:
                    delta = chunk["choices"][0].get("delta", {})
                    if "content" in delta:
                        content = delta["content"]
                        full_response += content
                        await websocket.send_json({
                            "type": "stream",
                            "content": content
                        })
                        await asyncio.sleep(0.01)

            await websocket.send_json({
                "type": "done",
                "content": full_response
            })

    except WebSocketDisconnect:
        manager.disconnect(websocket)
    except Exception as e:
        print(f"Error: {e}")
        await websocket.close()


if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host="127.0.0.1",
        port=8080,
        reload=True,
        reload_dirs=[".", "templates"]
    )