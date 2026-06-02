import os
import torch
import gradio as ui
from llama_cpp import Llama
from huggingface_hub import hf_hub_download

hf_hub_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q5_0-00001-of-00002.gguf"
)

hf_hub_download(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q5_0-00002-of-00002.gguf"
)


# ==================== 환경 설정 ====================
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)
os.environ["HF_TOKEN"] = ""
# ==================== Llama.cpp 모델 로드 ====================
model_id = "Qwen/Qwen2.5-7B-Instruct-GGUF"
filename="qwen2.5-7b-instruct-q5_k_m.gguf"

print(f"Llama.cpp로 모델 로드 중... (RTX 5070 윈도우 최적화)")

llm = Llama.from_pretrained(
    repo_id="Qwen/Qwen2.5-7B-Instruct-GGUF",
    filename="qwen2.5-7b-instruct-q5_0-00001-of-00002.gguf",
    n_gpu_layers=-1,
    n_ctx=8192,
    verbose=False
)

print("✅ Llama.cpp 모델 로드 완료!")

# ==================== Gradio 챗봇 ====================
def bot_turn(chat_history):
    if not chat_history:
        return chat_history

    # Llama.cpp의 create_chat_completion 포맷에 맞게 메시지 구성
    messages = [{"role": "system", "content": "당신은 친절하고 똑똑한 AI 개인 비서입니다. 한국어로 자연스럽게 답변해주세요."}]

    for msg in chat_history:
        if msg["content"]:
            messages.append({"role": msg["role"], "content": msg["content"]})

    # Llama.cpp는 스트리밍(글자가 실시간으로 나오는 효과)을 지원하므로 스트리밍으로 구현
    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        repeat_penalty=1.05,
        stream=True  # 실시간 글자 출력 활성화
    )

    chat_history[-1]["content"] = ""
    for chunk in response:
        delta = chunk['choices'][0]['delta']
        if 'content' in delta:
            chat_history[-1]["content"] += delta['content']
            yield chat_history


def user_turn(user_message, chat_history):
    chat_history.append({"role": "user", "content": user_message})
    chat_history.append({"role": "assistant", "content": ""})
    return "", chat_history


# ==================== Gradio UI ====================
with ui.Blocks() as demo:
    ui.Markdown("# 💬 Qwen2.5-7B AI")

    chatbot = ui.Chatbot(label="대화창", type="messages", height=600)

    with ui.Row():
        msg = ui.Textbox(
            label="메시지 입력",
            placeholder="무엇이든 물어보세요...",
            scale=9
        )
        clear = ui.Button("대화 초기화", scale=1)

    msg.submit(
        user_turn,
        inputs=[msg, chatbot],
        outputs=[msg, chatbot],
        queue=False
    ).then(
        bot_turn,
        inputs=chatbot,
        outputs=chatbot
    )

    clear.click(lambda: [], None, chatbot, queue=False)

if __name__ == "__main__":
    demo.launch(share=True)