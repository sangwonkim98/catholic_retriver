# [담당자] Person 2: 검색 리드
# [핵심 임무] 사용자 질문(query)을 받아 가장 관련성 높은 문서 청크를 반환하는 `retrieve` 함수를 제공합니다.
# [입출력 계약]
#   - retrieve(query: str, k: int) -> list[dict]
#   - Output 형태: [{"chunk_id": str, "chunk_text": str}, ...]
# [To-Do]
#   1. 초기에는 Mock 데이터를 반환하여 다른 모듈이 의존성 없이 개발할 수 있도록 합니다.
#   2. `init_retriever` 함수에서 실제 FAISS 인덱스와 텍스트 맵을 로드하는 로직을 구현합니다.
#   3. `retrieve` 함수를 실제 FAISS 검색 로직으로 교체합니다.
#   4. (Advanced) Cross-encoder를 사용한 리랭킹 로직을 추가합니다.

import os
import json
import faiss
from sentence_transformers import SentenceTransformer

# --- 초기화 ---
# [To-Do] 아래 Mock 초기화 부분을 실제 인덱스 로딩 코드로 교체해야 합니다.
# 예: index = faiss.read_index(...)
#     with open(...) as f: chunks_map = json.load(f)
#     model = SentenceTransformer(...)

FAISS_INDEX = None
CHUNKS_MAP = None
EMBEDDING_MODEL = None

def init_retriever():
    """
    FAISS 인덱스, 텍스트 맵, 임베딩 모델을 로드하여 리트리버를 초기화합니다.
    [To-Do] 앱 시작 시 한 번만 호출되도록 설계해야 합니다.
    """
    global FAISS_INDEX, CHUNKS_MAP, EMBEDDING_MODEL
    print("리트리버 초기화를 시작합니다 (현재는 Mock 모드).")
    # [To-Do] 실제 파일 로드 로직 구현
    # DB_PATH = os.path.join(os.path.dirname(__file__), '..', 'db')
    # INDEX_PATH = os.path.join(DB_PATH, "textbook.index")
    # MAP_PATH = os.path.join(DB_PATH, "chunks_map.json")
    # if os.path.exists(INDEX_PATH) and os.path.exists(MAP_PATH):
    #     FAISS_INDEX = faiss.read_index(INDEX_PATH)
    #     with open(MAP_PATH, 'r', encoding='utf-8') as f:
    #         CHUNKS_MAP = json.load(f)
    #     EMBEDDING_MODEL = SentenceTransformer('jhgan/ko-sbert-nli')
    #     print("실제 인덱스와 모델을 로드했습니다.")
    # else:
    #     print("경고: 인덱스 파일이 없어 Mock 모드로 작동합니다.")
    pass

def retrieve(query: str, k: int = 3) -> list[dict]:
    """
    사용자 질문과 가장 유사한 k개의 문서 청크를 검색합니다.
    """
    # [To-Do] 아래 Mock 코드를 실제 FAISS 검색 로직으로 교체해야 합니다.
    if FAISS_INDEX is None or CHUNKS_MAP is None or EMBEDDING_MODEL is None:
        print("Mock 검색 결과를 반환합니다.")
        return [
            {"chunk_id": "mock_chunk_1", "chunk_text": "이것은 첫 번째 가짜(mock) 검색 결과입니다. 강아지 예방접종에 대한 내용입니다."},
            {"chunk_id": "mock_chunk_2", "chunk_text": "이것은 두 번째 가짜(mock) 검색 결과입니다. 고양이 체온에 관한 정보입니다."},
            {"chunk_id": "mock_chunk_3", "chunk_text": "이것은 세 번째 가짜(mock) 검색 결과입니다. 반려동물과 초콜릿의 위험성에 대해 다룹니다."}
        ][:k]
    
    # # --- 실제 검색 로직 (구현 시 주석 해제) ---
    # print(f"'{query}'에 대한 실제 검색을 수행합니다.")
    # query_embedding = EMBEDDING_MODEL.encode([query], convert_to_tensor=False)
    # distances, indices = FAISS_INDEX.search(query_embedding.astype(np.float32), k)
    
    # # chunk_id는 0부터 시작하는 정수 인덱스라고 가정
    # all_chunk_ids = list(CHUNKS_MAP.keys())
    # results = []
    # for i in range(len(indices[0])):
    #     chunk_idx = indices[0][i]
    #     chunk_id = all_chunk_ids[chunk_idx]
    #     results.append({
    #         "chunk_id": chunk_id,
    #         "chunk_text": CHUNKS_MAP[chunk_id]
    #     })
    # return results

# 스크립트 로드 시 초기화 함수 호출 (테스트용)
# 실제 앱에서는 app.py에서 명시적으로 호출하는 것이 좋습니다.
init_retriever()

if __name__ == '__main__':
    # 모듈 단독 테스트
    test_query = "강아지 예방접종 언제부터 해야해?"
    retrieved_docs = retrieve(test_query, k=2)
    print(f"
--- 테스트 검색 결과 (Query: {test_query}) ---")
    print(json.dumps(retrieved_docs, indent=4, ensure_ascii=False))
