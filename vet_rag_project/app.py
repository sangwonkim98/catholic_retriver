# app.py
# [담당자] Person 4 / 통합 리드
# 목적: Gradio UI 구동 + RAG 파이프라인 오케스트레이션
#
# 파이프라인:
#   User query
#     → module_search.retrieve()
#     → module_augment.build_prompt()
#     → module_model.generate_answer()
#     → UI 출력 (답변 + 근거 요약)
#
# [입출력 계약]
#   - 입력: 사용자의 한국어 질문(str)
#   - 출력: 최종 답변(str) (+ 하단에 근거 섹션을 함께 Markdown으로 렌더)
#
# 불변식 / 합의:
#   - 앱 시작 시 init_retriever() 1회 호출하여 인덱스/맵/모델을 메모리에 로드
#   - module_search.retrieve(query, top_k, score_threshold) 사용(파라미터명 top_k로 통일)
#   - score는 0~1 스케일(정규화된 IP 기반) → UI 컷오프 설정 가능
#
# To-Do:
#   - argparse/환경변수로 top_k, score_threshold, 모델명 등 파라미터화
#   - 로그 수집/보안 헤더 추가
#   - 스트리밍 출력(원하면 ChatInterface 대신 Blocks로 전환)

from __future__ import annotations

import os
import traceback
from typing import List, Dict

import gradio as gr

# --- 모듈 임포트 ---
from src.module_search import retrieve, init_retriever
from src.module_augment import build_prompt
from src.module_model import generate_answer

# =========================
# UI 헬퍼: 근거 포맷터
# =========================
def _format_evidence(context_docs: List[Dict]) -> str:
    """
    검색된 문서들(context_docs)을 사람이 보기 좋은 Markdown으로 요약.
    내부 규약:
      - doc: {"chunk_id": int, "chunk_text": str, "score": float(0~1), "meta": {...}}
    """
    if not context_docs:
        return "> 근거가 비어 있습니다."

    lines = []
    for i, d in enumerate(context_docs, start=1):
        meta = d.get("meta", {}) or {}
        title = meta.get("title", "N/A")
        dept = meta.get("department", "N/A")
        score = d.get("score", 0.0)
        txt = d.get("chunk_text", "").strip().replace("\n", " ")
        if len(txt) > 280:
            txt = txt[:280] + "…"
        lines.append(f"- [{i}] (id={d.get('chunk_id', '-')}, score={score:.2f}) {title} / {dept}\n  - {txt}")
    body = "\n".join(lines)
    return f"**참고 근거**\n\n{body}"

# =========================
# RAG 파이프라인
# =========================
def rag_pipeline(query: str, top_k: int = 5, score_threshold: float = 0.55) -> str:
    """
    사용자 질문에 대해 RAG 파이프라인을 실행하고 최종 답변을 Markdown 문자열로 반환.

    파라미터:
      - top_k: 검색 상위 몇 개를 사용할지
      - score_threshold: 0~1 컷오프. 이보다 낮은 근거는 제외
    """
    q = (query or "").strip()
    if not q:
        return "질문이 비어 있습니다. 예: '강아지 예방접종은 언제부터 하나요?'처럼 입력해 주세요."

    try:
        # 1) 검색
        docs = retrieve(q, top_k=top_k, score_threshold=score_threshold, return_meta=True)
        # 2) 프롬프트 구성
        prompt = build_prompt(q, docs)
        # 3) 생성
        answer = generate_answer(prompt)

        # 4) 응답 패키징(답변 + 근거 섹션)
        evidence_md = _format_evidence(docs)
        final = f"{answer}\n\n---\n{evidence_md}"
        return final

    except Exception as e:
        # 예외는 사용자에게 깔끔히 안내 + 서버 로그로 남김
        traceback.print_exc()
        return f"죄송합니다. 처리 중 오류가 발생했습니다.\n\n세부: {e}"

# =========================
# Gradio UI
# =========================
def setup_ui():
    """
    ChatInterface 기반의 간단한 UI.
    - 더 고급 기능(파라미터 슬라이더 등)은 Blocks로 확장 가능.
    """
    examples = [
        "강아지 예방접종은 언제부터 시작하나요?",
        "고양이 정상 체온은 얼마인가요?",
        "개가 초콜릿을 먹었는데 어떻게 해야 하나요?",
    ]

    def _on_message(message, history):
        # history는 사용하지 않고, message만 받아서 처리
        return rag_pipeline(message, top_k=5, score_threshold=0.55)

    iface = gr.ChatInterface(
        fn=_on_message,
        title="수의학 RAG 챗봇",
        description="질문을 입력하면, 인덱스에서 관련 근거를 찾아 답변합니다. 근거는 하단에 요약 표기됩니다.",
        examples=examples,
        theme="soft",
    )
    return iface

# =========================
# 엔트리포인트
# =========================
if __name__ == "__main__":
    # 앱 시작 시 1회 초기화: 인덱스/맵/모델 메모리 로드
    # 인덱스/맵이 없거나 로드 실패 시, module_search는 Mock로 폴백
    print("[app] 리트리버 초기화 시작")
    init_retriever()  # 기본 경로(db/textbook.index, db/chunks_map.json)와 모델명으로 로드

    print("[app] Gradio UI 기동")
    ui = setup_ui()
    # 공유 링크 필요 없으면 share=False
    ui.launch(share=True)