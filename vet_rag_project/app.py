# [담당자] Person 4 / 통합 리드
# [핵심 임무] Gradio UI를 실행하고, 각 모듈(검색, 증강, 모델)을 순서대로 호출하여 RAG 파이프라인을 완성합니다.
# [입출력 계약]
#   - 사용자 입력(Query) -> RAG 파이프라인 -> 최종 답변(Response)
# [To-Do]
#   1. `init_retriever()`를 앱 시작 시 한 번만 호출하여 모델/인덱스를 로드하도록 구현합니다.
#   2. `chat_function` 내에서 RAG 파이프라인(retrieve -> build_prompt -> generate_answer)을 완성합니다.
#   3. (Optional) 답변과 함께 근거가 된 문서를 UI에 함께 표시하는 기능을 추가합니다.

import gradio as gr

# --- 모듈 임포트 ---
# 각 팀원이 개발한 모듈의 핵심 함수들을 가져옵니다.
from src.module_search import retrieve, init_retriever
from src.module_augment import build_prompt
from src.module_model import generate_answer

# --- RAG 파이프라인 정의 ---
def rag_pipeline(query: str) -> str:
    """
    사용자 질문에 대해 RAG 파이프라인을 실행하여 최종 답변을 반환합니다.
    """
    print(f"\n--- 새로운 질문 --- \nQuery: {query}")
    
    # 1. 검색 (Retrieve)
    # [To-Do] k값은 최적화를 통해 조절할 수 있습니다.
    retrieved_docs = retrieve(query, k=3)
    print("1. [검색 완료]")
    print(retrieved_docs)
    
    # 2. 프롬프트 구성 (Augment)
    prompt = build_prompt(query, retrieved_docs)
    print("\n2. [프롬프트 생성 완료]")
    print(prompt)

    # 3. 답변 생성 (Generate)
    answer = generate_answer(prompt)
    print("\n3. [답변 생성 완료]")
    print(answer)
    
    # [To-Do] 근거 문서(retrieved_docs)를 답변과 함께 예쁘게 포맷하여 반환할 수 있습니다.
    # formatted_answer = f"""{answer}
    #
    # ---
    # **참고 자료:**
    # { "\n".join([doc['chunk_text'] for doc in retrieved_docs]) }
    # """
    # return formatted_answer
    
    return answer

# --- Gradio UI 설정 ---
def setup_ui():
    """
    Gradio 인터페이스를 설정하고 실행합니다.
    """
    interface = gr.ChatInterface(
        fn=rag_pipeline,
        title="🐾 수의학 RAG 챗봇",
        description="궁금한 점을 질문해주세요. 예: '강아지 예방접종은 언제부터 하나요?'",
        examples=["강아지 예방접종 언제부터 해야해?", "고양이 정상 체온은?", "강아지가 초콜릿을 먹으면 어떻게 돼?"],
        theme="soft",
    )
    return interface

if __name__ == "__main__":
    # --- 애플리케이션 시작 ---
    # [To-Do] 앱 시작 시 리트리버(FAISS 인덱스, 임베딩 모델 등)를 미리 로드합니다.
    # 이렇게 하면 매번 질문할 때마다 로드하지 않아 시간이 단축됩니다.
    print("애플리케이션을 시작합니다...")
    # init_retriever() # 실제 인덱스 로딩이 구현되면 주석 해제
    
    app_ui = setup_ui()
    app_ui.launch(share=True)
