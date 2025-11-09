# build_index.py
# [담당] Person 1 — 인덱스 구축 리드
# 목적: JSONL → 청킹 → 임베딩 → FAISS(index) + chunks_map.json 생성
# 사용 시나리오:
#   - 오프라인 사전 단계에서만 실행한다. (데이터가 바뀔 때마다)
#   - 생성물은 온라인 단계(module_search.retrieve)에서 불러 쓴다.
# 핵심 불변식(invariants):
#   - chunks_map[i]의 i는 곧 FAISS ID다. (리스트 순서 == 인덱스 추가 순서)
#   - 검색 단계에서 사용할 임베딩 모델과 여기서 쓴 모델은 반드시 동일해야 한다.
#   - normalize=True면 FAISS는 IndexFlatIP(내적)로 만들어 코사인 유사도를 근사한다.


# [입력 JSONL 한 줄 예시: data/.../*.jsonl]
# ---------------------------------------------
# {"title": "소동물 주요 질환의 임상추론과 감별진단", 
#  "author": "현창백 내과아카데미 역", 
#  "publisher": "(주)범문에듀케이션", 
#  "department": "내과", 
#  "disease": "\n\n황달 증례의 임상적 추론\n황달 증례에 대한 임상적 분석\n7\n황달 증례의 임상적 추론\n황달 증례에 대한 임상적 분석\n임상 증례\n황달\nV는 4살의 ... (이하 긴 본문) ... 고빌리루빈혈 ////"
# }

# $ python src/build_index.py
#  - build_index.py: main() →  load_and_chunk_data() → create_and_save_index()
#
#  - BASE_DIR/data/**/*.jsonl 아래 파일을 모두 스캔한다.
#  - text_field="disease" 값을 꺼내어 청킹 기준("\n\n")으로 분할한다.

# ─────────────────────────────────────────────
# 1) load_and_chunk_data() 단계
# ─────────────────────────────────────────────
#  (1) _iter_jsonl()이 jsonl 파일을 한 줄씩 읽고 dict로 yield.
#      JSON 오류 라인은 건너뜀.
#
#  (2) 각 라인에서 text_field="disease" 값을 뽑고,
#      _chunk_text(text, sep="\n\n", max_chunk_chars=1200, min_len=30)
#      로 문단 단위 청킹을 수행.
#
#  (3) 청킹 결과를 순서대로 append → 이 "순서"가 곧 FAISS ID가 된다.
#      (불변식: chunks_map[i]의 i == 인덱스 추가 순서 == FAISS ID)
#
#  예) 위 한 줄에서 disease 본문이 "\n\n" 기준으로 아래처럼 나뉘었다고 가정:
#      (아래는 길이 제한/공백 정리 후의 "정제된" 문단 일부 샘플)
#
#   parts = [
#     "황달 증례의 임상적 추론\n황달 증례에 대한 임상적 분석\n7\n황달 증례의 임상적 추론\n황달 증례에 대한 임상적 분석",
#     "임상 증례\n황달\nV는 4살의 중성화한 수컷 프렌치 불독으로 7일 동안 무감각증(apathy), 식욕저하, 구토 증상을 보였으며 지난 24시간 동안 황달 소견이 관찰되었다. ...",
#     "사례에 적용되는 추론 방법",
#     "확보된 정보를 바탕으로 다음 2단계의 임상적 추론을 진행한다:\n1. 환자의 문제를 파악한다\n2. 문제의 우선순위를 설정한다\n환자는 지난 7일 동안 식욕부진, ...",
#     "... (이하 여러 문단) ...",
#     "황달은 빌리루빈 축적으로 인한 혈관 내 빌리루빈 수치 증가, 즉 고빌리루빈혈 ..."
#   ]
#
#  (4) 각 문단을 아래 포맷으로 리스트에 쌓는다:
#   chunks = [
#     {
#       "chunk_id": 0,
#       "chunk_text": "황달 증례의 임상적 추론\n황달 증례에 대한 임상적 분석\n7\n황달 증례의 임상적 추론\n황달 증례에 대한 임상적 분석",
#       "meta": {"title": "소동물 주요 질환의 임상추론과 감별진단", "department": "내과"}
#     },
#     {
#       "chunk_id": 1,
#       "chunk_text": "임상 증례\n황달\nV는 4살의 ... (긴 문단; 필요 시 1200자에서 컷)",
#       "meta": {"title": "소동물 주요 질환의 임상추론과 감별진단", "department": "내과"}
#     },
#     {
#       "chunk_id": 2,
#       "chunk_text": "사례에 적용되는 추론 방법",
#       "meta": {"title": "소동물 주요 질환의 임상추론과 감별진단", "department": "내과"}
#     },
#     {
#       "chunk_id": 3,
#       "chunk_text": "확보된 정보를 바탕으로 다음 2단계의 임상적 추론을 진행한다:\n1. 환자의 문제를 파악한다\n2. 문제의 우선순위를 설정한다\n환자는 지난 7일 동안 ...",
#       "meta": {"title": "소동물 주요 질환의 임상추론과 감별진단", "department": "내과"}
#     },
#     ...
#     {
#       "chunk_id": N,
#       "chunk_text": "황달은 빌리루빈 축적으로 인한 혈관 내 빌리루빈 수치 증가, 즉 고빌리루빈혈 ...",
#       "meta": {"title": "소동물 주요 질환의 임상추론과 감별진단", "department": "내과"}
#     }
#   ]
#
#  ※ 같은 파일/같은 라인에서 여러 문단이 나오면 순서대로 ID가 부여됨.
#  ※ 여러 파일을 처리하는 경우, 이전 파일의 마지막 ID 다음 번호부터 이어서 증가.

# ─────────────────────────────────────────────
# 2) create_and_save_index() 단계
# ─────────────────────────────────────────────
#  (1) 임베딩 모델 로드: SentenceTransformer("jhgan/ko-sbert-nli")
#      - 이 모델명은 "검색 단계(module_search)"에서도 "동일하게" 사용해야 한다.
#        (임베딩 공간이 달라지면 유사도 계산이 무의미해짐)
#
#  (2) texts = [c["chunk_text"] for c in chunks]
#      - 길이: len(chunks)
#
#  (3) _encode_in_batches(model, texts, batch_size=256, normalize=True)
#      - 결과: embeddings.shape == (len(chunks), D)
#              (D는 보통 768 차원; 모델에 따라 다름)
#      - normalize=True → 각 벡터를 L2 정규화(코사인 유사도 근사)
#
#  (4) FAISS 인덱스 생성
#      - normalize=True 이므로 IndexFlatIP(내적) 사용 → 코사인 유사도 근사로 동작
#      - index.add(embeddings.astype(np.float32))
#
#  (5) 디스크 저장
#      - INDEX_PATH = DB_DIR / "textbook.index"
#        → 벡터 인덱스(바이너리). 사람이 읽을 수 없음.
#      - MAP_PATH   = DB_DIR / "chunks_map.json"
#        → 위에서 만든 chunks 리스트를 그대로 JSON으로 저장.
#
#  (6) 저장물의 예시(일부; chunks_map.json):
# [
#   {
#     "chunk_id": 0,
#     "chunk_text": "황달 증례의 임상적 추론\n황달 증례에 대한 임상적 분석\n7\n황달 증례의 임상적 추론\n황달 증례에 대한 임상적 분석",
#     "meta": {"title": "소동물 주요 질환의 임상추론과 감별진단", "department": "내과"}
#   },
#   {
#     "chunk_id": 1,
#     "chunk_text": "임상 증례\n황달\nV는 4살의 ...",
#     "meta": {"title": "소동물 주요 질환의 임상추론과 감별진단", "department": "내과"}
#   },
#   ...
# ]
#
#  ※ 리스트이기 때문에, "리스트 인덱스 == FAISS ID" 라는 불변식이 보장된다.

# ─────────────────────────────────────────────
# 3) 온라인 단계(module_search.retrieve)에서의 소비 방식
# ─────────────────────────────────────────────
#  - init_retriever():
#       index  = faiss.read_index("db/textbook.index")
#       chunks = json.load(open("db/chunks_map.json"))
#       model  = SentenceTransformer("jhgan/ko-sbert-nli")  # 반드시 동일 모델
#
#  - retrieve(query, top_k):
#       q_vec = model.encode([query], normalize_embeddings=True)  # 빌드 규약과 동일 설정
#       scores, ids = index.search(q_vec.astype(np.float32), top_k)
#       # ids 예: [[15, 1, 3, 0, 9]]
#       # → 각 id에 대해 chunks_map[id]를 읽고, 
#       #    {"chunk_id": id, "chunk_text": ..., "meta": ..., "score": 변환된 점수}
#       #   형태로 반환.
#
#  - 점수 스케일:
#       IndexFlatIP + 정규화된 벡터 → 내적값이 [-1, 1] 범위(코사인 유사도).
#       UI/컷오프 용도로 0~1로 간단히 변환할 수도 있음 (예: (ip+1)/2).

# ─────────────────────────────────────────────
# 4) 왜 두 파일이 분리되어 저장될까?
# ─────────────────────────────────────────────
#  - textbook.index: "숫자 벡터"만 효율적으로 저장/검색하는 용도 (FAISS 바이너리)
#  - chunks_map.json: "사람이 읽을 원문과 메타"를 보관하는 용도
#    → 검색 ID → 원문 매핑이 이 파일로 가능
#    → 원문/메타만 바꾸고 싶을 때, 인덱스 재빌드 없이 교체는 원칙적으론 불가(임베딩 재계산 필요).
#      (본문이 바뀌면 벡터가 바뀌므로 반드시 재빌드 권장)
#
#  - 분리 저장의 장점:
#     * 검색 속도 최적화(FAISS는 순수 벡터에 특화)
#     * 원문/메타를 JSON으로 투명하게 관리(버저닝/검증/리뷰 용이)

# ─────────────────────────────────────────────
# 5) 운영/튜닝 팁
# ─────────────────────────────────────────────
#  - 청킹 규칙:
#     * sep="\n\n" → 문단 기준. 책/리포트류에 적합.
#     * sep=". "   → 문장 기준. 과도한 쪼개짐으로 맥락 상실 가능; overlap 고려.
#     * 겹치기(overlap) 도입: 이전/다음 문장 일부를 포함시켜 문맥 유지.
#
#  - 길이 제한:
#     * max_chunk_chars=1200: 프롬프트 토큰 초과 방지용 안전장치.
#     * 데이터 특성/모델 콘텍스트 창에 맞춰 800~1400 사이에서 팀 합의.
#
#  - 모델/메트릭 일관성:
#     * 빌드와 검색에서 같은 임베딩 모델명과 normalize 설정을 사용.
#     * normalize=True ↔ IndexFlatIP(코사인 근사)
#       normalize=False ↔ IndexFlatL2
#
#  - 재빌드 트리거:
#     * data/*.jsonl 변경(추가/수정/삭제) 시 반드시 재빌드.
#     * 청킹 규칙 변경 시에도 재빌드(벡터가 전부 달라짐).

import os
import json
import glob
from typing import List, Dict, Iterable

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# =========================
# 상수 & 기본 경로 정의
#  - BASE_DIR를 기준으로 상대 경로 혼란을 없앤다.
#  - 경로를 바꾸면 검색 모듈의 로딩 경로도 동일하게 바꿔야 한다.
# =========================
BASE_DIR = Path(__file__).resolve().parent.parent   # 프로젝트 루트 기준 ../
DATA_GLOB = str(BASE_DIR / "data" / "**" / "*.jsonl")  # 하위 폴더 포함 모든 jsonl
DB_DIR = BASE_DIR / "db"
INDEX_PATH = DB_DIR / "textbook.index"             # FAISS 인덱스 바이너리
MAP_PATH = DB_DIR / "chunks_map.json"              # 청크 리스트(JSON)
EMBEDDING_MODEL = "jhgan/ko-sbert-nli"             # 검색 단계와 동일해야 함- 텍스트임베딩 모델 

# =========================
# 유틸: 텍스트 청킹
#  - 기본은 문단 단위("\n\n")로 자른다.
#  - 너무 짧은 조각은 제거하고, 너무 긴 조각은 잘라서 모델 컨텍스트 초과를 방지.
#  - 필요하면 문장 분할기로 교체 가능(팀 합의 후).
# =========================
def _chunk_text( text: str,
 sep: str = "\n\n",          # 튜닝 포인트: 문단 ↔ 문장(". ") 등으로 변경 가능
    max_chunk_chars: int = 1200,# 튜닝 포인트: 모델 컨텍스트/프롬프트 길이에 맞춰 조정(800~1400 권장)
    min_len: int = 30,          # 튜닝 포인트: 노이즈 제거 하한
) -> List[str]:
    
    """
    텍스트를 sep 기준으로 나누고 공백/빈문자 제거 후 길이 필터 적용.
    너무 긴 청크는 max_chunk_chars로 절단.

    확장 아이디어:
      - 문장 토크나이저 기반 청킹(KoNLPy/kiwi 등)
      - 슬라이딩 윈도우 겹치기(overlap)로 문맥 보존
      - 표/리스트 전처리(마크업 보존)
    """
    raw_chunks = [c.strip() for c in text.split(sep)]
    clean = []
    for c in raw_chunks:
        if not c:
            continue                  # 빈 조각 제외
        if len(c) < min_len:
            continue                  # 너무 짧은 조각 제외(잡음)
        if len(c) > max_chunk_chars:  # 너무 긴 조각은 절단(토크나이저 없이 안전한 컷)
            c = c[:max_chunk_chars]
        clean.append(c)
    return clean

def _iter_jsonl(file_path: str) -> Iterable[dict]:
    """
    jsonl 라인을 제너레이터로 제공하여 메모리 사용량을 낮춘다.
    JSON 구문 오류 라인은 조용히 건너뛴다.
    """
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError:
                continue

# =========================
# 1) 로드 & 청킹
#  - 입력: jsonl 여러 개
#  - 출력: [{"chunk_id": int, "chunk_text": str, "meta": {...}}, ...]
#  - 실패:
#      * 파일 0개 → FileNotFoundError
#      * 유효 청크 0개 → ValueError
# =========================
def load_and_chunk_data(
    glob_path: str = DATA_GLOB,
    text_field: str = "disease",   # 데이터 본문이 담긴 키. 필요 시 "content" 등으로 교체
    chunk_sep: str = "\n\n",
    max_chunk_chars: int = 1200,
    min_len: int = 30,
) -> List[Dict]:
    """
    JSONL 데이터 로드 → 텍스트 청킹 → 메타 포함 리스트 생성.

    출력 형식 예:
      [
        {
          "chunk_id": 0,                         # 0..N-1 (FAISS ID와 동일)
          "chunk_text": "문단/문장 텍스트 ...",
          "meta": {
            "title": "개(2판)-...",
            "department": "안과"
          }
        },
        ...
      ]
    """
    file_paths = glob.glob(glob_path, recursive=True)
    if not file_paths:
        raise FileNotFoundError(f"'{glob_path}' 경로에서 .jsonl 파일을 찾지 못했어요.")

    chunks: List[Dict] = []
    cid = 0

    print(f"[load] 파일 {len(file_paths)}개 스캔 시작")
    for fp in file_paths:
        for obj in _iter_jsonl(fp):
            # 본문 텍스트 추출
            text = str(obj.get(text_field, "")).strip()
            if not text:
                continue
            # 청킹(문단/문장 등)
            parts = _chunk_text(
                text, sep=chunk_sep, max_chunk_chars=max_chunk_chars, min_len=min_len
            )
            for p in parts:
                chunks.append(
                    {
                        "chunk_id": cid,     # 리스트 순서 == FAISS ID 불변식
                        "chunk_text": p,
                        "meta": {
                            "title": obj.get("title", "N/A"),
                            "department": obj.get("department", "N/A"),
                        },
                    }
                )
                cid += 1

    if not chunks:
        # 보통 text_field 오타, sep가 지나치게 촘촘, min_len이 너무 큰 경우
        raise ValueError("유효한 청크가 1개도 생성되지 않았어요.")

    print(f"[chunk] 총 {len(chunks)}개 청크 생성")
    return chunks

# =========================
# 2) 임베딩 & 인덱스 저장
#  - normalize=True면 IndexFlatIP(내적)로 코사인 유사도 근사
#  - normalize=False면 IndexFlatL2(유클리드 거리)
#  - 메모리 절약을 위해 배치 인코딩 사용
# =========================
def _encode_in_batches(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = 256,
    normalize: bool = True
) -> np.ndarray:
    """
    대용량 텍스트 임베딩을 배치로 처리.
    배치 크기는 GPU/메모리 상황에 맞춰 조정.
    """
    all_vecs: List[np.ndarray] = []
    for i in range(0, len(texts), batch_size):
        batch = texts[i : i + batch_size]
        vecs = model.encode(batch, normalize_embeddings=normalize)
        all_vecs.append(np.asarray(vecs, dtype=np.float32))
    return np.vstack(all_vecs)

def create_and_save_index(
    chunks: List[Dict],
    index_path: Path = INDEX_PATH,
    map_path: Path = MAP_PATH,
    embedding_model: str = EMBEDDING_MODEL,
    batch_size: int = 256,
    normalize: bool = True,
    use_inner_product: bool = True,
) -> Dict:
    """
    텍스트 청크 → 임베딩 → FAISS 인덱스 + 청크맵 저장.

    반환:
      {"num_chunks": int, "index_path": str, "map_path": str,
       "embedding_model": str, "normalize": bool, "metric": "inner_product"|"l2"}

    실패:
      - chunks 비어 있음 → ValueError
      - 인덱스/맵 저장 실패 → RuntimeError
    """
    if not chunks:
        raise ValueError("청킹된 데이터가 비어 있어요.")

    DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[embed] 모델 로드: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    texts = [c["chunk_text"] for c in chunks]
    print(f"[embed] 임베딩 시작 (개수={len(texts)}, batch={batch_size})")
    embeddings = _encode_in_batches(model, texts, batch_size=batch_size, normalize=normalize)

    # 메트릭 선택: normalize=True ↔ IP, normalize=False ↔ L2
    dim = embeddings.shape[1]
    if use_inner_product and normalize:
        index = faiss.IndexFlatIP(dim)   # 코사인 근사(정규화된 벡터)
    elif not use_inner_product and not normalize:
        index = faiss.IndexFlatL2(dim)   # L2 거리
    else:
        # 설정 상충 시 자동 보정. 운영 시에는 둘 중 하나로 일관되게 쓰자.
        print("[warn] normalize/use_inner_product 조합 보정 → IP 사용")
        index = faiss.IndexFlatIP(dim)

    print(f"[faiss] add vectors → {index_path}")
    index.add(embeddings.astype(np.float32))

    # 인덱스 저장
    try:
        faiss.write_index(index, str(index_path))
    except Exception as e:
        raise RuntimeError(f"FAISS 인덱스 저장 실패: {e}")

    # 청크 맵 저장(리스트 형태, 인덱스 == FAISS ID)
    try:
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"청크 맵 저장 실패: {e}")

    summary = {
        "num_chunks": len(chunks),
        "index_path": str(index_path),
        "map_path": str(map_path),
        "embedding_model": embedding_model,
        "normalize": normalize,
        "metric": "inner_product" if isinstance(index, faiss.IndexFlatIP) else "l2",
    }
    print(f"[done] {summary}")
    return summary

# =========================
# 3) 엔트리포인트
#  - 실제 운영에선 argparse로 파라미터화 권장
#  - 로그 파일 저장, 진행률 표시 등은 팀 필요에 따라 추가
# =========================
def main():
    """
    파이프라인:
      1) load_and_chunk_data
      2) create_and_save_index
    """
    print("인덱스 빌드를 시작할게요.")
    chunks = load_and_chunk_data(
        glob_path=DATA_GLOB,     # 데이터 폴더 구조 변경 시 여기만 바꾸면 됨
        text_field="disease",    # 본문 키명. 실데이터에 맞게 조정
        chunk_sep="\n\n",        # 문단 기준. 필요 시 문장 기준 등으로 변경
        max_chunk_chars=1200,
        min_len=30,
    )
    create_and_save_index(
        chunks,
        index_path=INDEX_PATH,
        map_path=MAP_PATH,
        embedding_model=EMBEDDING_MODEL,
        batch_size=256,
        normalize=True,          # 검색 단계와 동일하게 유지
        use_inner_product=True,  # normalize=True와 세트
    )

if __name__ == "__main__":
    main()