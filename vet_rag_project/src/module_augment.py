# [담당자] Person 3: 증강 리드
# [핵심 임무] 사용자 질문(query)과 검색된 문서(context)를 조합하여 LLM에 전달할 최종 프롬프트(prompt)를 생성합니다.
# [입출력 계약]
#   - build_prompt(query: str, context_docs: list[dict]) -> str
#   - Input `context_docs` 형태: [{"chunk_id": str, "chunk_text": str}, ...]
# [To-Do]
#   1. 시스템 프롬프트, Few-shot 예시, CoT(Chain-of-Thought) 등 다양한 프롬프트 엔지니어링 기법을 실험하여 최적의 프롬프트를 설계합니다.
#   2. 검색된 근거(context)를 효과적으로 프롬프트에 삽입하는 방법을 연구합니다.

def build_prompt(query: str, context_docs: list[dict]) -> str:
    """
    검색된 문서를 바탕으로 LLM에 전달할 프롬프트를 생성합니다.
    """
    # 1. 검색된 근거(context)를 하나의 문자열로 조합
    context_str = "\n\n".join([doc["chunk_text"] for doc in context_docs])

    # 2. 시스템 프롬프트 정의
    # [To-Do] 이 부분을 지속적으로 개선하여 LLM의 답변 품질을 높여야 합니다.
    system_prompt = """당신은 반려동물에 대한 지식을 바탕으로 질문에 답변하는 전문 수의사 AI입니다.
주어진 '근거'를 바탕으로, 사용자의 '질문'에 대해 친절하고 명확하게 답변해주세요.
만약 '근거'에 답변의 내용이 없다면, '제공된 정보만으로는 답변하기 어렵습니다.'라고 솔직하게 답변하세요.
답변은 반드시 한국어로 작성해야 합니다."""

    # 3. 사용자 질문과 근거를 결합하여 최종 프롬프트 생성
    # CoT(Chain-of-Thought)를 유도하기 위해 생각의 단계를 포함시킬 수 있습니다.
    user_prompt = f"""[근거]
{context_str}

[질문]
{query}

[답변]
위 '근거'를 바탕으로 '질문'에 대해 답변을 생성해 보겠습니다.
"""

    # 최종 프롬프트 조합 (예시: 시스템 프롬프트와 사용자 프롬프트를 분리하여 전달하는 경우)
    # 실제 API 호출 시에는 API 스펙에 맞춰 조합 방식이 달라질 수 있습니다.
    # 여기서는 간단히 하나의 문자열로 합칩니다.
    final_prompt = f"{system_prompt}\n\n{user_prompt}"
    
    return final_prompt

if __name__ == '__main__':
    # 모듈 단독 테스트
    test_query = "강아지 예방접종 언제부터 해야해?"
    test_context = [
        {"chunk_id": "mock_chunk_1", "chunk_text": "강아지는 생후 6주부터 예방접종을 시작해야 합니다. 종합백신(DHPPL), 코로나 장염, 켄넬코프, 광견병 예방접종이 필요합니다."},
        {"chunk_id": "mock_chunk_2", "chunk_text": "새끼 강아지의 사회화 시기는 생후 3주에서 12주 사이가 매우 중요합니다."}
    ]
    
    prompt = build_prompt(test_query, test_context)
    print("--- 생성된 프롬프트 ---")
    print(prompt)
