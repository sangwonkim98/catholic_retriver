import json
import os
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import glob # 하위 폴더 검색용

# --- 상수 정의 ---
# [수정] 님의 데이터 구조에 맞게 경로 수정
DATA_PATH = "../data/**/*.jsonl" # 'data' 폴더 하위의 모든 .jsonl 검색
DB_PATH = "../db"
INDEX_PATH = os.path.join(DB_PATH, "textbook.index")
MAP_PATH = os.path.join(DB_PATH, "chunks_map.json")
EMBEDDING_MODEL = 'jhgan/ko-sbert-nli'

def load_and_chunk_data(glob_path)
    """
    JSONL 데이터를 로드하고, 의미 기반(\n\n)으로 텍스트를 청킹합니다.
    [수정] 님의 실제 데이터 필드("disease")와 청킹 전략(\n\n)을 반영합니다.
    """
    chunks_with_metadata = [] # [핵심] 청크와 메타데이터를 저장할 '리스트'
    chunk_id_counter = 0

    file_paths = glob.glob(glob_path, recursive=True)
    if not file_paths:
        print(f"오류: '{glob_path}' 경로에서 .jsonl 파일을 찾을 수 없습니다.")
        return []

    print(f"총 {len(file_paths)}개의 .jsonl 파일 로드 시작...")

    for file_path in file_paths:
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    data = json.loads(line.strip())
                    
                    # [수정] 'content' -> 'disease' (지난번 확인한 님의 필드명)
                    text = data.get('disease', '') 
                    if not text:
                        continue
                        
                    # [수정] '. ' -> '\n\n' (문단 기준 청킹)
                    raw_chunks = text.split('\n\n')
                    
                    for chunk_text in raw_chunks:
                        chunk_text = chunk_text.strip()
                        if chunk_text:
                            # [핵심] 청크 맵은 리스트이며, '순번'이 곧 FAISS ID가 됩니다.
                            chunks_with_metadata.append({
                                "chunk_id": chunk_id_counter, # FAISS ID(순번)
                                "chunk_text": chunk_text,    # 원본 텍스트
                                "source_title": data.get("title", "N/A"), # 메타데이터
                                "source_dept": data.get("department", "N/A") # 메타데이터
                            })
                            chunk_id_counter += 1
                except json.JSONDecodeError:
                    continue # JSON 오류 라인 무시

    print(f"총 {len(chunks_with_metadata)}개의 청크를 생성했습니다.")
    return chunks_with_metadata # 딕셔너리가 아닌 '리스트' 반환

def create_and_save_index(chunks_with_metadata):
    """
    텍스트 청크를 임베딩하고 FAISS 인덱스와 텍스트 맵(List)을 저장합니다.
    """
    if not chunks_with_metadata:
        print("청킹된 데이터가 없습니다. 인덱스 생성을 건너뜁니다.")
        return

    print("임베딩 모델을 로드합니다...")
    model = SentenceTransformer(EMBEDDING_MODEL)
    
    print("텍스트 임베딩을 시작합니다...")
    # [수정] 'chunk_text' 필드만 뽑아서 임베딩
    texts = [item['chunk_text'] for item in chunks_with_metadata]
    embeddings = model.encode(texts, normalize_embeddings=True)
    
    print("FAISS 인덱스를 생성하고 저장합니다...")
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(np.array(embeddings, dtype=np.float32))
    
    if not os.path.exists(DB_PATH):
        os.makedirs(DB_PATH)
        
    faiss.write_index(index, INDEX_PATH)
    
    # --- [핵심 수정] ---
    # 딕셔너리({key: value})가 아닌, '리스트' 자체를 저장합니다.
    # Person 2는 FAISS에서 반환된 숫자 '5'를
    # 이 리스트의 5번째 인덱스(chunks_map[5])로 조회할 수 있습니다.
    print("청크 맵 (List)을 저장합니다...")
    with open(MAP_PATH, 'w', encoding='utf-8') as f:
        json.dump(chunks_with_metadata, f, ensure_ascii=False, indent=4)

    print("인덱스 및 청크 맵 생성이 완료되었습니다.")

if __name__ == "__main__":
    print("인덱스 빌드를 시작합니다...")
    # [수정] DATA_PATH 변경
    chunked_data = load_and_chunk_data(DATA_PATH) 
    create_and_save_index(chunked_data)