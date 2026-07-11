# BILLIARD COACH

당구대 사진(전체/일부)을 찍으면 자동으로 탑뷰로 변환해 샷 가이드를 제공하는 앱.
기획서: [billiard_coach_MVP_기획서.md](billiard_coach_MVP_기획서.md)

## 구성

| 폴더 | 내용 |
|---|---|
| `server/` | Python FastAPI — 다이아몬드 검출 + 호모그래피 탑뷰 변환 |
| `app/` | Flutter Android 앱 |

## 서버 실행 (외부 접속 포함)

**간단 실행**: `C:\coding\billiard\start_server.bat` 더블클릭 → 창 2개가 열림
(서버 + Cloudflare 터널). 터널 창에 표시되는 `https://xxxx.trycloudflare.com`
주소를 앱 인트로 톱니바퀴에 입력하면 어디서든(당구장 포함) 접속 가능.

주의사항:
- **PC가 켜져 있고 두 창이 떠 있는 동안만** 앱이 동작함
- 터널 주소는 재시작할 때마다 바뀜 → 그때마다 앱에서 주소 갱신 필요
- 고정 주소가 필요하면 `server/Dockerfile`로 Render/Railway/Fly.io 등에
  배포 (호스팅 계정 필요)

수동 실행:
```powershell
cd C:\coding\billiard\server
$env:PYTHONIOENCODING='utf-8'
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
# 별도 창에서:
& 'C:\Program Files (x86)\cloudflared\cloudflared.exe' tunnel --url http://localhost:8000
```

## 서버 테스트 / CLI

```powershell
cd C:\coding\billiard\server
.venv\Scripts\python.exe -m pytest tests -q            # 전체 테스트
.venv\Scripts\python.exe tools\run_topview.py 사진.jpg  # 사진 1장 탑뷰 변환
.venv\Scripts\python.exe tools\make_synthetic.py       # 합성 테스트 이미지 생성
```

## 앱 빌드 / 설치

```powershell
$env:JAVA_HOME = 'C:\Program Files\Microsoft\jdk-17.0.19.10-hotspot'
cd C:\coding\billiard\app
C:\src\flutter\bin\flutter.bat analyze
C:\src\flutter\bin\flutter.bat test
C:\src\flutter\bin\flutter.bat build apk --debug
& "$env:LOCALAPPDATA\Android\Sdk\platform-tools\adb.exe" install -r build\app\outputs\flutter-apk\app-debug.apk
```

## 진행 상황 (기획서 5장 개발 순서)

- [x] 1. 정지 이미지 다이아몬드 검출 + 호모그래피 변환 (합성 이미지 검증: 20/20, 오차 0.9mm)
- [x] 2. 실시간 검출 + 피드백 (B안) — 프리뷰 프레임을 서버로 보내 분석 (개발용; 추후 온디바이스 전환 검토)
- [x] 3. 공 4개 위치 검출 (합성 검증: 4/4 색 일치, 위치 오차 20mm 이내)
- [x] 4. 기본 샷 패턴 룰 기반 가이드 (당점 보정 ±40° 분리각 모델, 키스 등급)
- [x] 5. 샷 패턴 라이브러리 — 일반(직접/1쿠션/2쿠션 걸어치기) + 3쿠션(뒤/앞/옆돌려치기 등, 거울 반사 모델) 탭 분리
- [x] 6. 공 움직임 애니메이션 (플레이 버튼)
- [~] 7. 메인화면 UI/UX — 카드 캐러셀 + 당점 스나이퍼 + 두께 다이어그램 (시점 회전·가로 모드는 미착수)
