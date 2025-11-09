# module_search.py
# [담당자] Person 2 — 검색 리드
# 목적: 사용자 질의를 받아 FAISS에서 유사 청크를 검색하여 반환
# 사용 시나리오: 온라인 단계(app.py → retrieve(query))
#
# 핵심 불변식(invariants)
#   - chunks_map[i]의 i == FAISS ID (build_index.py에서 리스트 순서로 저장)
#   - 임베딩 모델/정규화 설정은 build_index.py와 동일해야 함
#     (우리는 normalize=True, IndexFlatIP 가정)
#
# I/O Contract
#   init_retriever(index_path, map_path, embedding_model, normalize) -> None
#   retrieve(query: str, top_k: int = 5, score_threshold: float|None = None) -> list[dict]
#     반환 예:
#       [
#         {
#           "chunk_id": 123,
#           "chunk_text": "...",
#           "score": 0.87,                 # 0~1 스케일(정규화된 IP를 0~1로 변환)
#           "meta": {"title": "...", "department": "..."}
#         }, ...
#       ]
#
# 실패/폴백
#   - 인덱스/맵/모델 로드 실패 또는 미초기화 → Mock 결과 반환(개발속도 확보)
#   - 빈 질의/공백 → ValueError

from __future__ import annotations

import os
import json
from typing import List, Dict, Optional, Tuple
from pathlib import Path

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

# =========================
# 경로/기본 설정 (build_index.py와 일치)
# =========================
BASE_DIR = Path(__file__).resolve().parent  # src/
ROOT_DIR = BASE_DIR.parent                  # 프로젝트 루트
DB_DIR = ROOT_DIR / "db"
DEFAULT_INDEX_PATH = DB_DIR / "textbook.index"
DEFAULT_MAP_PATH = DB_DIR / "chunks_map.json"
DEFAULT_EMBED_MODEL = "jhgan/ko-sbert-nli"

# 검색 설정(빌드와 일치)
NORMALIZE = True          # build_index: normalize=True
USE_INNER_PRODUCT = True  # build_index: IndexFlatIP

# =========================
# 전역 핸들(앱 시작 시 1회 로드)
# =========================
_FAISS_INDEX: Optional[faiss.Index] = None
_CHUNKS_MAP: Optional[List[Dict]] = None
_EMBED_MODEL: Optional[SentenceTransformer] = None
_LOADED_PATHS: Tuple[str, str, str] | None = None  # (index_path, map_path, embed_model_name)

# =========================
# 내부 유틸
# =========================
def _raw_to_unit_score(raw: float) -> float:
    """
    정규화된 임베딩 + IP인 경우 raw 점수는 [-1, 1] 범위.
    UI/로직 편의상 0~1로 변환하여 반환.
    """
    # 보수적으로 클램프
    return float(max(0.0, min(1.0, (raw + 1.0) / 2.0)))

def _paths_exist(index_path: Path, map_path: Path) -> bool:
    return index_path.exists() and map_path.exists()

# =========================
# 1) 초기화
# =========================
def init_retriever(
    index_path: str | Path = DEFAULT_INDEX_PATH,
    map_path: str | Path = DEFAULT_MAP_PATH,
    embedding_model_name: str = DEFAULT_EMBED_MODEL,
    normalize: bool = NORMALIZE,
) -> None:
    """
    FAISS 인덱스, 청크 맵, 임베딩 모델을 메모리에 1회 로드.
    앱 시작 시(app.py) 호출 권장.

    파라미터:
      - index_path/map_path: build_index 산출물 경로
      - embedding_model_name: build_index와 동일 모델
      - normalize: 쿼리 임베딩 정규화 여부(인덱스 메트릭과 일치)

    실패 시:
      - 예외를 던지지 않고 경고 출력 후 Mock 모드로 유지(개발 생산성 우선)
    """
    global _FAISS_INDEX, _CHUNKS_MAP, _EMBED_MODEL, _LOADED_PATHS

    try:
        index_path = Path(index_path)
        map_path = Path(map_path)

        if not _paths_exist(index_path, map_path):
            print(f"[search:init] 경고: 인덱스/맵이 없습니다. Mock 모드로 동작합니다. "
                  f"(index={index_path}, map={map_path})")
            _FAISS_INDEX = None
            _CHUNKS_MAP = None
            _EMBED_MODEL = None
            _LOADED_PATHS = None
            return

        print(f"[search:init] 인덱스 로드: {index_path}")
        _FAISS_INDEX = faiss.read_index(str(index_path))

        print(f"[search:init] 청크 맵 로드: {map_path}")
        with open(map_path, "r", encoding="utf-8") as f:
            _CHUNKS_MAP = json.load(f)

        print(f"[search:init] 임베딩 모델 로드: {embedding_model_name}")
        _EMBED_MODEL = SentenceTransformer(embedding_model_name)

        _LOADED_PATHS = (str(index_path), str(map_path), embedding_model_name)
        print(f"[search:init] 완료 (chunks={len(_CHUNKS_MAP)})")

    except Exception as e:
        # 개발 단계 편의: 예외를 올리지 않고 Mock으로 폴백
        print(f"[search:init] 경고: 초기화 실패 → Mock 모드 전환. 이유: {e}")
        _FAISS_INDEX = None
        _CHUNKS_MAP = None
        _EMBED_MODEL = None
        _LOADED_PATHS = None

def _ensure_ready() -> bool:
    """리트리버가 실제 검색 가능한 상태인지 검사."""
    return _FAISS_INDEX is not None and _CHUNKS_MAP is not None and _EMBED_MODEL is not None

# =========================
# 2) 검색
# =========================
def retrieve(
    query: str,
    top_k: int = 5,
    score_threshold: Optional[float] = None,  # 0~1 스케일 컷오프(예: 0.6)
    return_meta: bool = True,
) -> List[Dict]:
    """
    사용자 질의에 대해 상위 top_k 유사 청크를 반환.

    제약:
      - query는 공백 제외 1자 이상
      - 초기화 실패/미실행 시 Mock 결과 반환(개발 편의)

    반환:
      - [{"chunk_id": int, "chunk_text": str, "score": float(0~1), "meta": {...}}, ...]
        (return_meta=False면 meta 키 생략)

    점수 스케일:
      - build_index와 동일 가정(정규화+IP) → raw∈[-1,1]를 0~1로 선형변환하여 반환
    """
    q = (query or "").strip()
    if not q:
        raise ValueError("query가 비어있어요. 최소 한 글자 이상 입력해 주세요.")

    # Mock 폴백
    if not _ensure_ready():
        print("[search] Mock 검색 결과를 반환합니다.")
        base = [
            {"chunk_id": 1, "chunk_text": "강아지 예방접종은 보통 생후 6~8주부터 시작합니다.", "score": 0.88},
            {"chunk_id": 2, "chunk_text": "고양이 정상 체온은 대략 38.1~39.2℃ 범위입니다.", "score": 0.72},
            {"chunk_id": 3, "chunk_text": "초콜릿의 테오브로민은 반려동물에 독성이 있습니다.", "score": 0.69},
        ]
        return base[:top_k]

    # 실제 검색
    try:
        # 1) 쿼리 임베딩
        q_emb = _EMBED_MODEL.encode([q], normalize_embeddings=NORMALIZE)
        q_vec = np.asarray(q_emb, dtype=np.float32)

        # 2) 유사도 검색
        #   - IP 기준: 값이 클수록 유사
        #   - L2 기준: 값이 작을수록 유사(하지만 우리는 IP 가정)
        raw_scores, raw_ids = _FAISS_INDEX.search(q_vec, top_k)

        results: List[Dict] = []
        ids = raw_ids[0].tolist()
        scores = raw_scores[0].tolist()

        for chunk_idx, raw in zip(ids, scores):
            # 안전판: 인덱스 범위 확인
            if chunk_idx < 0 or chunk_idx >= len(_CHUNKS_MAP):
                continue

            item = _CHUNKS_MAP[chunk_idx]
            # 점수 변환(정규화된 IP → 0~1)
            s = _raw_to_unit_score(raw) if USE_INNER_PRODUCT and NORMALIZE else float(raw)

            if score_threshold is not None and s < score_threshold:
                continue

            one = {
                "chunk_id": int(item.get("chunk_id", chunk_idx)),
                "chunk_text": item.get("chunk_text", ""),
                "score": float(s),
            }
            if return_meta:
                one["meta"] = item.get("meta", {})
            results.append(one)

        return results

    except Exception as e:
        # 장애 시 Mock 폴백
        print(f"[search] 경고: 검색 중 오류 → Mock 폴백. 이유: {e}")
        return [
            {"chunk_id": -1, "chunk_text": "검색 중 오류가 발생했어요. 질문을 다시 시도해 주세요.", "score": 0.0}
        ]

# =========================
# 모듈 단독 테스트
# =========================
if __name__ == "__main__":
    # 앱 구동 전 수동 테스트 시:
    # 1) 오프라인 빌드가 완료되어 db/* 파일이 있어야 실제 검색이 동작
    # 2) 없으면 자동으로 Mock 결과 반환
    init_retriever(
        index_path=DEFAULT_INDEX_PATH,
        map_path=DEFAULT_MAP_PATH,
        embedding_model_name=DEFAULT_EMBED_MODEL,
        normalize=NORMALIZE,
    )

    q = "강아지 예방접종 언제부터 해야 해?"
    print(f"[test] query = {q}")
    docs = retrieve(q, top_k=5, score_threshold=0.55)
    print(json.dumps(docs, ensure_ascii=False, indent=2))