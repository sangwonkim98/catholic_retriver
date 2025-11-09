# build_index.py
# [담당] Person 1 — 인덱스 구축 리드
# 목적: JSON(단일 객체/리스트) → 청킹 → 임베딩 → FAISS(index) + chunks_map.json 생성
# 사용 시나리오:
#   - 오프라인 사전 단계에서만 실행 (데이터가 바뀔 때마다 재빌드)
#   - 산출물(textbook.index, chunks_map.json)은 온라인 단계(module_search.retrieve)에서 로드
# 핵심 불변식(invariants):
#   - chunks_map[i]의 i == FAISS ID (리스트 순서 == 인덱스 추가 순서)
#   - 검색 단계의 임베딩 모델/정규화 설정과 반드시 동일해야 함
#   - normalize=True ↔ IndexFlatIP(내적, 코사인 근사), normalize=False ↔ IndexFlatL2

from __future__ import annotations

import os
import re
import io
import json
import glob
from typing import List, Dict, Iterable

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from pathlib import Path

# ─────────────────────────────────────────────
# 0) 경로/모델 상수
#    - DATA_DIR: 기본은 프로젝트 루트의 ./data
#      (환경변수 RAG_DATA_DIR 로 재지정 가능)
# ─────────────────────────────────────────────
BASE_DIR = Path(__file__).resolve().parent.parent        # 예) vet_rag_project/
DATA_DIR = Path(os.getenv("RAG_DATA_DIR", BASE_DIR / "data"))
DB_DIR   = BASE_DIR / "db"

INDEX_PATH = DB_DIR / "textbook.index"                   # FAISS 인덱스(바이너리)
MAP_PATH   = DB_DIR / "chunks_map.json"                  # 청크 리스트(JSON)

EMBEDDING_MODEL = "jhgan/ko-sbert-nli"                   # 검색 단계와 동일해야 함
NORMALIZE_EMB   = True                                   # True면 IP(코사인 근사) 사용
BATCH_SIZE      = 256                                    # 임베딩 배치 크기

# ─────────────────────────────────────────────
# 1) JSON 로더 (jsonl 제외, json만 지원)
#    - 파일 하나가 단일 객체(dict)거나 리스트(list[dict])인 경우만 지원
# ─────────────────────────────────────────────
def _iter_json(file_path: str) -> Iterable[Dict]:
    """
    JSON 파일 전체를 읽어 dict 또는 list[dict]를 yield.
    (jsonl은 지원하지 않음)
    """
    with io.open(file_path, "r", encoding="utf-8") as f:
        try:
            obj = json.load(f)
        except json.JSONDecodeError:
            return
        
        #     obj (로드시 딕셔너리로 열림)    ##{
        #   "title": "소동물 주요 질환의 임상추론과 감별진단",
        #   "department": "내과",
        #   "disease": "황달 증례의 임상적 추론..."
        # } 
    if isinstance(obj, dict):
        yield obj
    elif isinstance(obj, list):
        for o in obj:
            if isinstance(o, dict):
                yield o

# ─────────────────────────────────────────────
# 2) 텍스트 청킹
#    - 기본: 두 줄 이상 개행(빈 줄) 기준 분리 (r"\n{2,}")
#    - 너무 짧은/긴 조각 필터링
#    - 필요 시 겹치기(overlap) 도입 가능 (TODO)
# ─────────────────────────────────────────────
def _chunk_text(
    text: str,
    max_chunk_chars: int = 1200,
    min_len: int = 30,
) -> List[str]:
    """
    개행 정규화 후 '빈 줄(두 줄 이상 개행)' 기준으로 문단을 분리.
    너무 짧은/긴 문단은 필터링.
    """
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    raw_chunks = [c.strip() for c in re.split(r"\n{2,}", text)]
    clean: List[str] = []
    for c in raw_chunks:
        if not c or len(c) < min_len:
            continue
        if len(c) > max_chunk_chars:
            c = c[:max_chunk_chars]
        clean.append(c)
    return clean

# ─────────────────────────────────────────────
# 3) 로드 & 청킹
#    - 입력: ./data/**.json (하위 폴더 포함)
#    - 출력: [{"chunk_id": int, "chunk_text": str, "meta": {...}}, ...]
# ─────────────────────────────────────────────
def load_and_chunk_data(
    data_dir: Path = DATA_DIR,         # 데이터 폴더 기본값: 프로젝트 루트의 /data
    text_field: str = "disease",       # JSON 안에서 본문 텍스트가 들어있는 key
    max_chunk_chars: int = 1200,       # 문단당 최대 길이 제한 (너무 길면 자름)
    min_len: int = 30,                 # 너무 짧은 문단은 제외
) -> List[Dict]:
    """
    JSON 파일들을 스캔 → 본문 추출 → 문단 단위 청킹 → (FAISS ID 포함) 리스트 생성
    실패:
      - 파일 0개 → FileNotFoundError
      - 유효 청크 0개 → ValueError
    """
    # ------------------------------------------------------------------
    # glob.glob()은 파일 경로를 자동으로 탐색하는 함수야.
    # - str(data_dir / "**" / "*.json")
    #   → 예: '.../data/**/*.json' 형태로 만들어짐
    #   → data 폴더 안의 모든 하위폴더(** 포함)에서
    #      확장자가 .json인 파일을 전부 찾는다.
    #   → 결과: ['data/내과/강아지.json', 'data/외과/고양이.json', ...]
    # ------------------------------------------------------------------
    pattern = str(data_dir / "**" / "*.json")
    file_paths = glob.glob(pattern, recursive=True)  # recursive=True → 하위폴더까지 탐색

    if not file_paths:
        raise FileNotFoundError(f"[load] 입력 JSON이 없습니다: {pattern}")

    chunks: List[Dict] = []   # 결과 리스트 (모든 문단 조각을 담음)
    cid = 0                   # 각 청크에 고유 ID 부여 (== FAISS 인덱스 ID)

    print(f"[load] 파일 {len(file_paths)}개 스캔 시작 (data_dir={data_dir})")

    # ------------------------------------------------------------------
    # 파일 하나씩 순회하며 내용 추출
    #   _iter_json(fp) → 파일 안의 dict들을 하나씩 yield (한 JSON 파일에 여러 dict 가능)
    # ------------------------------------------------------------------
    for fp in file_paths:
        for obj in _iter_json(fp):
            text = str(obj.get(text_field, "")).strip()   # disease 필드 내용 꺼냄
            if not text:
                continue

            # _chunk_text() → 문단 단위로 쪼갠 문자열 리스트 반환
            # (빈 줄 두 개 기준으로 분리, 너무 짧거나 긴 건 제외)
            parts = _chunk_text(text, max_chunk_chars=max_chunk_chars, min_len=min_len)

            # 각 문단을 하나의 “청크”로 등록
            for p in parts:
                chunks.append(
                    {
                        "chunk_id": cid,   # 이 번호가 곧 FAISS index의 ID (중요한 불변식)
                        "chunk_text": p,   # 실제 텍스트 조각
                        "meta": {          # 부가 정보: 검색 결과에 함께 보여줄 수 있음
                            "title": obj.get("title", "N/A"),
                            "department": obj.get("department", "N/A"),
                        },
                    }
                )
                cid += 1  # 다음 문단에는 다음 ID 부여

    # ------------------------------------------------------------------
    # 유효 청크가 하나도 없으면 오류 발생
    # ------------------------------------------------------------------
    if not chunks:
        raise ValueError("[chunk] 유효한 청크가 1개도 생성되지 않았습니다.")

    print(f"[chunk] 총 {len(chunks)}개 청크 생성")
    return chunks

# ─────────────────────────────────────────────
# 4) 임베딩 (배치) & 인덱스 저장
#    - NORMALIZE_EMB=True → IndexFlatIP (코사인 근사)
#    - NORMALIZE_EMB=False → IndexFlatL2
# ─────────────────────────────────────────────
def _encode_in_batches(
    model: SentenceTransformer,
    texts: List[str],
    batch_size: int = BATCH_SIZE,
    normalize: bool = NORMALIZE_EMB,
) -> np.ndarray:
    """대용량 텍스트 임베딩을 배치로 처리."""
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
    normalize: bool = NORMALIZE_EMB,
) -> Dict:
    """
    텍스트 청크 → 임베딩 → FAISS 인덱스 + 청크맵 저장.
    반환 요약 dict는 로그/테스트 용도.
    """
    if not chunks:
        raise ValueError("[embed] 입력 chunks가 비었습니다.")

    DB_DIR.mkdir(parents=True, exist_ok=True)

    print(f"[embed] 모델 로드: {embedding_model}")
    model = SentenceTransformer(embedding_model)

    texts = [c["chunk_text"] for c in chunks]
    print(f"[embed] 임베딩 시작 (개수={len(texts)}, batch={BATCH_SIZE}, normalize={normalize})")
    embeddings = _encode_in_batches(model, texts, batch_size=BATCH_SIZE, normalize=normalize)
    
    dim = embeddings.shape[1]
    if normalize:
        index = faiss.IndexFlatIP(dim)     # 정규화된 벡터 → IP == 코사인 근사
    else:
        index = faiss.IndexFlatL2(dim)

    print(f"[faiss] add vectors → {index_path}")
    index.add(embeddings.astype(np.float32))

    # 인덱스 저장
    try:
        faiss.write_index(index, str(index_path))
    except Exception as e:
        raise RuntimeError(f"[faiss] 인덱스 저장 실패: {e}")

    # 청크 맵(JSON) 저장 — 리스트 인덱스 == FAISS ID
    try:
        with open(map_path, "w", encoding="utf-8") as f:
            json.dump(chunks, f, ensure_ascii=False, indent=2)
    except Exception as e:
        raise RuntimeError(f"[map] 청크 맵 저장 실패: {e}")

    summary = {
        "num_chunks": len(chunks),
        "index_path": str(index_path),
        "map_path": str(map_path),
        "embedding_model": embedding_model,
        "normalize": normalize,
        "metric": "inner_product" if normalize else "l2",
    }
    print(f"[done] {summary}")
    return summary

# ─────────────────────────────────────────────
# 5) 엔트리포인트
#    - 실제 운영에선 argparse 붙여 파라미터화 권장
# ─────────────────────────────────────────────
def main():
    """
    파이프라인:
      1) load_and_chunk_data
      2) create_and_save_index
    """
    print("[start] 인덱스 빌드를 시작합니다.")
    chunks = load_and_chunk_data(
        data_dir=DATA_DIR,
        text_field="disease",    # 데이터 본문 키명 (필요 시 변경 가능)
        max_chunk_chars=1200,
        min_len=30,
    )
    create_and_save_index(
        chunks,
        index_path=INDEX_PATH,
        map_path=MAP_PATH,
        embedding_model=EMBEDDING_MODEL,
        normalize=NORMALIZE_EMB,
    )

if __name__ == "__main__":
    main()