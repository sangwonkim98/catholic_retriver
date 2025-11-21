
import sys
import subprocess
import json
import glob
import argparse
from tqdm.auto import tqdm

# --- 유틸리티 함수 ---
def install_package(package):
    """필요한 파이썬 패키지를 설치합니다."""
    print(f"필요한 라이브러리 '{package}'를 설치합니다.")
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package],
                              stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except subprocess.CalledProcessError as e:
        print(f"'{package}' 설치 중 오류 발생: {e}")
        sys.exit(1)

# --- 1. 문장 단위 청킹 ---
def perform_sentence_chunking(data_path, max_length=500):
    """KSS를 사용하여 문장 단위로 텍스트를 청킹합니다."""
    try:
        import kss
    except ImportError:
        install_package("kss")
        import kss

    json_files = glob.glob(f"{data_path}/**/*.json", recursive=True)
    print(f"총 {len(json_files)}개의 JSON 파일에서 문장 단위 청킹을 시작합니다.")

    all_chunks = []
    for file_path in tqdm(json_files, desc="문장 단위 청킹"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text = data.get("disease", "")
            if not text:
                continue

            sentences = kss.split_sentences(text)
            
            current_chunk = ""
            for sentence in sentences:
                if len(current_chunk) + len(sentence) > max_length and current_chunk:
                    all_chunks.append({
                        "content": current_chunk.strip(),
                        "source": data.get("title", "N/A"),
                        "department": data.get("department", "N/A")
                    })
                    current_chunk = ""
                
                current_chunk += " " + sentence
            
            if current_chunk:
                all_chunks.append({
                    "content": current_chunk.strip(),
                    "source": data.get("title", "N/A"),
                    "department": data.get("department", "N/A")
                })
        except (json.JSONDecodeError, IOError) as e:
            print(f"파일 처리 오류: {file_path}, 오류: {e}")
            
    return all_chunks

# --- 2. 의미 기반 청킹 ---
def perform_semantic_chunking(data_path, chunk_size=500, chunk_overlap=50):
    """RecursiveCharacterTextSplitter를 사용하여 의미 단위로 텍스트를 청킹합니다."""
    try:
        from langchain_text_splitters import RecursiveCharacterTextSplitter
    except ImportError:
        install_package("langchain-text-splitters")
        from langchain_text_splitters import RecursiveCharacterTextSplitter

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
        length_function=len,
    )

    json_files = glob.glob(f"{data_path}/**/*.json", recursive=True)
    print(f"총 {len(json_files)}개의 JSON 파일에서 의미 기반 청킹을 시작합니다.")

    all_chunks = []
    for file_path in tqdm(json_files, desc="의미 기반 청킹"):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            text = data.get("disease", "")
            if not text:
                continue

            chunks = text_splitter.split_text(text)
            
            for chunk_content in chunks:
                all_chunks.append({
                    "content": chunk_content,
                    "source": data.get("title", "N/A"),
                    "department": data.get("department", "N/A")
                })
        except (json.JSONDecodeError, IOError) as e:
            print(f"파일 처리 오류: {file_path}, 오류: {e}")
            
    return all_chunks

# --- 메인 실행 로직 ---
def main():
    parser = argparse.ArgumentParser(description="수의학 RAG 프로젝트를 위한 데이터 청킹 파이프라인")
    parser.add_argument(
        "--method",
        type=str,
        choices=["sentence", "semantic"],
        required=True,
        help="청킹 방법을 선택합니다: 'sentence' (문장 단위) 또는 'semantic' (의미 기반)"
    )
    parser.add_argument(
        "--data_path",
        type=str,
        default="../data",  # 스크립트 위치(vet_rag_project) 기준
        help="입력 데이터가 있는 디렉토리 경로"
    )
    parser.add_argument(
        "--output_file",
        type=str,
        default="chunks.jsonl",
        help="결과를 저장할 파일 이름 (JSONL 형식)"
    )
    args = parser.parse_args()

    if args.method == "sentence":
        chunks = perform_sentence_chunking(args.data_path)
    else: # args.method == "semantic"
        chunks = perform_semantic_chunking(args.data_path)

    # 결과를 JSONL 파일로 저장
    try:
        with open(args.output_file, 'w', encoding='utf-8') as f:
            for chunk in chunks:
                f.write(json.dumps(chunk, ensure_ascii=False) + '\n')
        
        print(f"\n✅ 총 {len(chunks)}개의 청크가 생성되어 '{args.output_file}' 파일에 저장되었습니다.")
        
    except IOError as e:
        print(f"결과 파일 저장 중 오류 발생: {e}")

if __name__ == "__main__":
    main()
