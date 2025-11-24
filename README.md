RAG 웹 앱 실행을 위한 최종 가이드

  요청하신 모든 코드(build_index.py, app.py, web_app.py) 작성이 완료되었습니다. 아래 단계에
  따라 직접 RAG 챗봇 웹 앱을 실행해보실 수 있습니다.

  1. 라이브러리 설치

  웹 앱에 필요한 gradio를 포함하여, 프로젝트에 필요한 모든 라이브러리를 설치합니다. 터미널을
  열고 아래 명령어를 입력하세요.

   1 pip install -r vet_rag_project/requirements.txt

  2. OpenAI API 키 설정

   1. 프로젝트의 최상위 폴더(rag 폴더)에 .env 라는 이름의 새 파일을 만드세요.
   2. 방금 만든 .env 파일을 열고, 발급받은 OpenAI API 키를 아래 형식으로 붙여넣으세요.

   1     OPENAI_API_KEY="sk-..."

  3. 데이터베이스 생성 (인덱싱)

  웹 앱을 실행하기 전에, 질문에 답변할 때 참고할 지식 데이터베이스를 만들어야 합니다. 이 과정은
  데이터 양에 따라 시간이 걸릴 수 있습니다.

  터미널에서 아래 명령어를 실행하세요.

   1 python vet_rag_project/build_index.py

  성공적으로 완료되면 vet_rag_project/db/ 폴더 안에 vector_store.index와 chunks_map.json 파일이
  생성됩니다.

  4. 웹 앱 실행

  이제 모든 준비가 끝났습니다. 아래 명령어를 실행하여 웹 앱을 시작하세요.

   1 python vet_rag_project/web_app.py

  5. 웹 앱 접속

  웹 앱이 실행되면 터미널에 아래와 비슷한 URL 주소가 나타납니다.

  Running on local URL: http://127.0.0.1:7860

  이 주소를 복사하여 크롬, 엣지 등 웹 브라우저의 주소창에 붙여넣으면, 귀여운 강아지 이미지가
  있는 챗봇 상담소를 사용하실 수 있습니다.



  수정 및 확장 포인트

   * 쿼리 재작성: app.py의 _rewrite_query 함수 안에 더 정교한 쿼리 재작성 로직을 추가할 수
     있습니다.
   * LLM 모델 변경: app.py의 Config 클래스에서 LLM_MODEL을 gpt-4-turbo 등 다른 모델로 변경할 수
     있습니다.
   * 청킹 방법 변경: build_index.py의 Config 클래스에서 CHUNKING_METHOD를 다른 방법으로 바꾸어
     여러 가지 청킹 전략을 실험해볼 수 있습니다. (단, SEMANTIC은 API 키가 필요합니다.)




  
