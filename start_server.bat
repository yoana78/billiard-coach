@echo off
rem BILLIARD COACH 서버 + 공개 터널 시작 스크립트
rem 실행하면 창 2개가 열립니다:
rem   1) 검출 서버 (닫으면 앱이 동작 안 함)
rem   2) Cloudflare 터널 — 창에 표시되는 https://xxxx.trycloudflare.com 주소를
rem      앱 인트로 화면의 톱니바퀴에 입력하세요 (주소는 실행할 때마다 바뀝니다)

start "BILLIARD SERVER" cmd /k "cd /d C:\coding\billiard\server && set PYTHONIOENCODING=utf-8 && .venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000"

timeout /t 3 /nobreak >nul

start "CLOUDFLARE TUNNEL" cmd /k ""C:\Program Files (x86)\cloudflared\cloudflared.exe" tunnel --url http://localhost:8000"
