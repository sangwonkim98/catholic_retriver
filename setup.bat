@echo off
echo [1/4] Conda 가상환경(vet_rag_env)을 생성합니다...
call conda create -n vet_rag_env python=3.10 -y

echo.
echo [2/4] 가상환경을 활성화합니다...
call conda activate vet_rag_env

echo.
echo [3/4] PyTorch (GPU 버전)를 먼저 설치합니다...
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu124

echo.
echo [4/4] 나머지 라이브러리를 설치합니다...
pip install -r requirements.txt

echo.
echo ========================================================
echo  설치가 완료되었습니다! 
echo  VS Code를 껐다 켜고 인터프리터를 'vet_rag_env'로 잡으세요.
echo ========================================================
pause