# -*- coding: utf-8 -*-
"""
[파일 역할]
이 스크립트는 RAG 파이프라인을 Gradio 웹 UI에 연결하여,
사용자가 웹 브라우저를 통해 AI와 채팅하며 반려동물에 대한 질문을 할 수 있도록 합니다.
`app.py`의 RAGPipeline을 가져와 사용합니다.

[실행 방법]
1. `build_index.py`를 실행하여 데이터베이스를 먼저 생성해야 합니다.
2. `.env` 파일에 OpenAI API 키가 설정되어 있어야 합니다.
3. 터미널에서 아래 명령어를 실행합니다.
   $ python vet_rag_project/web_app.py
4. 터미널에 표시된 URL (예: http://127.0.0.1:7860)을 웹 브라우저에서 열어 사용합니다.
"""
import gradio as gr
from app import RAGPipeline, Config

# --- 전역 설정 및 모델 로드 ---
# 웹 앱이 시작될 때 RAG 파이프라인을 한 번만 초기화합니다.
# 이렇게 하면 매번 질문할 때마다 모델을 새로 로드하지 않아 효율적입니다.
try:
    rag_pipeline = RAGPipeline(Config())
    print("[INFO] WebApp: RAG 파이프라인이 성공적으로 초기화되었습니다.")
except Exception as e:
    rag_pipeline = None
    print(f"[ERROR] WebApp: RAG 파이프라인 초기화 실패: {e}")
    # Gradio 인터페이스에서 이 오류를 사용자에게 알릴 것입니다.

# --- Gradio 인터페이스 함수 ---
def ask_ai_consultant(message: str, history: list) -> str:
    """
    Gradio 채팅 인터페이스와 RAG 파이프라인을 연결하는 함수.
    사용자 메시지를 받아 파이프라인을 실행하고 답변을 반환합니다.
    """
    if rag_pipeline is None:
        return "죄송합니다. RAG 파이프라인이 제대로 초기화되지 않았습니다. 터미널 로그를 확인해주세요."
    
    print(f"\n[Gradio] 사용자 질문 수신: {message}")
    # RAG 파이프라인 실행
    answer = rag_pipeline.run(message)
    print(f"[Gradio] AI 답변 생성: {answer}")
    
    return answer

# --- Gradio UI 구성 ---
with gr.Blocks(theme=gr.themes.Soft(), title="반려동물 AI 상담소") as web_app:
    gr.Markdown(
        """
        <div style="text-align: center; margin-bottom: 20px;">
            <img src="https://cdn.pixabay.com/photo/2020/06/30/22/04/dog-5357799_1280.png" style="width: 150px; margin: auto;">
            <h1 style="font-size: 24px;">반려동물 AI 상담소</h1>
            <p>궁금한 점을 무엇이든 물어보세요!</p>
        </div>
        """
    )
    
    gr.ChatInterface(
        fn=ask_ai_consultant,
        chatbot=gr.Chatbot(height=400),
        textbox=gr.Textbox(placeholder="예: 강아지가 자꾸 토해요. 원인이 뭘까요?", container=False, scale=7),
        title=None, # 위에서 직접 제목을 만들었으므로 None
        examples=["강아지가 갑자기 다리를 절어요", "고양이 피부에 각질이 생겼어요", "노견이 먹으면 좋은 영양제는?"],
        cache_examples=False,
    )

if __name__ == "__main__":
    print("[INFO] Gradio 웹 앱을 시작합니다.")
    web_app.launch()