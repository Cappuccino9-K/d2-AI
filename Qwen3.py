import os
import torch
import gradio as ui
from llama_cpp import Llama
from dotenv import load_dotenv

load_dotenv()

# ==================== 환경 설정 ====================
print("=" * 60)
print(f"PyTorch: {torch.__version__}")
print(f"CUDA: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
print("=" * 60)

# ==================== Llama.cpp 로컬 모델 로드 ====================
print("로컬에서 Qwen3-8B 모델 로드 중... (RTX 5070 최적화)")

# 💡 수정된 부분: 제공해주신 절대 경로와 파일명을 올바르게 매핑했습니다.
local_model_path = r"C:\Users\KYJ\PyCharmMiscProject\models\qwen\qwen3-8b-q5_k_m.gguf"

if not os.path.exists(local_model_path):
    print(f"❌ 에러: [{local_model_path}] 경로에서 모델 파일을 찾을 수 없습니다.")
    print("폴더명이나 파일명(qwen3-8b-q5_k_m.gguf)이 정확한지 다시 한 번 확인해주세요.")
    exit()

llm = Llama(
    model_path=local_model_path,  # 로컬 절대 경로 지정
    n_gpu_layers=-1,             # RTX 5070에서 전체 GPU 적재
    n_ctx=32768,                  # 필요시 16384로 증가 가능
    verbose=False,
)

print("✅ 로컬 Qwen3-8B-Q5_K_M 모델 로드 완료!")

# ==================== Gradio 챗봇 ====================
def bot_turn(chat_history):
    if not chat_history:
        return chat_history

    messages = [{"role": "system", "content": "당신은 친절하고 똑똑한 AI 개인 비서입니다. 한국어로 자연스럽게 답변해주세요."}]

    for msg in chat_history:
        if msg.get("content"):
            messages.append({"role": msg["role"], "content": msg["content"]})

    response = llm.create_chat_completion(
        messages=messages,
        temperature=0.7,
        top_p=0.9,
        max_tokens=1024,
        repeat_penalty=1.05,
        stream=True
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
    ui.Markdown("# 💬 Qwen3-8B AI (2025)")

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