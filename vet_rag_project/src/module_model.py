# [담당자] Person 4: 모델 리드
# [핵심 임무] 증강된 프롬프트(prompt)를 받아 LLM API를 호출하고, 답변을 생성하는 `generate_answer` 함수를 제공합니다.
# [입출력 계약]
#   - generate_answer(prompt: str) -> str
# [To-Do]
#   1. 초기에는 Mock 답변을 반환하여 다른 모듈이 의존성 없이 개발할 수 있도록 합니다.
#   2. `generate_answer` 함수에 실제 OpenAI API 호출 로직을 구현합니다.
#   3. API 응답에서 필요한 텍스트(content)만 추출하여 반환하도록 파싱 로직을 추가합니다.
#   4. API 키 관리를 안전하게 처리해야 합니다 (예: 환경 변수 사용).

import os
import openai

# --- 초기화 ---
# [To-Do] 절대 API 키를 코드에 하드코딩하지 마세요. os.getenv 등을 사용하세요.
# openai.api_key = os.getenv("OPENAI_API_KEY", "YOUR_OPENAI_API_KEY")
# 아래는 임시 설정이며, 실제 실행 전 반드시 유효한 키로 교체해야 합니다.
if "OPENAI_API_KEY" not in os.environ:
    print("경고: OPENAI_API_KEY 환경 변수가 설정되지 않았습니다. Mock 답변만 가능합니다.")
openai.api_key = os.environ.get("OPENAI_API_KEY", "sk-...")


def generate_answer(prompt: str) -> str:
    """
    LLM API를 호출하여 프롬프트에 대한 답변을 생성합니다.
    """
    # [To-Do] 아래 Mock 코드를 실제 OpenAI API 호출 로직으로 교체해야 합니다.
    is_mock = not os.environ.get("OPENAI_API_KEY") or os.environ.get("OPENAI_API_KEY") == "sk-..."

    if is_mock:
        print("Mock 답변을 생성합니다.")
        return "이것은 가짜(mock) 답변입니다. OpenAI API 연동이 필요합니다. 강아지는 생후 6주부터 접종을 시작하는 것이 좋습니다."

    # # --- 실제 API 호출 로직 (구현 시 주석 해제) ---
    # try:
    #     print("OpenAI API를 호출합니다...")
    #     response = openai.ChatCompletion.create(
    #         model="gpt-3.5-turbo",  # 또는 "gpt-4"
    #         messages=[
    #             {"role": "system", "content": "당신은 전문 수의사입니다."}, # Augment 모듈에서 프롬프트를 만들었다면 여기는 간단하게
    #             {"role": "user", "content": prompt}
    #         ],
    #         temperature=0.7,
    #         max_tokens=500,
    #     )
    #     # API 응답에서 답변 텍스트만 추출
    #     answer = response.choices[0].message['content']
    #     return answer
    # except Exception as e:
    #     print(f"OpenAI API 호출 중 오류 발생: {e}")
    #     return "죄송합니다, 답변을 생성하는 중 오류가 발생했습니다."


if __name__ == '__main__':
    # 모듈 단독 테스트
    test_prompt = """[근거]
강아지는 생후 6주부터 예방접종을 시작해야 합니다. 종합백신(DHPPL), 코로나 장염, 켄넬코프, 광견병 예방접종이 필요합니다.

[질문]
강아지 예방접종 언제부터 해야해?

[답변]
위 '근거'를 바탕으로 '질문'에 대해 답변을 생성해 보겠습니다.
"""
    answer = generate_answer(test_prompt)
    print("
--- 생성된 답변 ---")
    print(answer)
