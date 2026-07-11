# BILLIARD COACH

당구대 사진(전체/일부)을 찍으면 자동으로 탑뷰로 변환해 샷 가이드를 제공하는 앱.
기획서: [billiard_coach_MVP_기획서.md](billiard_coach_MVP_기획서.md)

## 구성

| 폴더 | 내용 |
|---|---|
| `server/` | Python FastAPI — 다이아몬드 검출 + 호모그래피 탑뷰 변환 |
| `app/` | Flutter Android 앱 |

## 서버

**프로덕션 (앱 기본값, PC 불필요)**: `https://billiard-coach.onrender.com`
- Render 클라우드, 코드는 GitHub `yoana78/billiard-coach`
- Free 플랜: 15분 유휴 시 슬립 → 첫 요청에 30초~1분 콜드스타트
  (앱이 시작할 때 미리 깨우는 요청을 보내 체감을 줄임)
- 서버 코드 반영: `git push` 후 **Render 대시보드에서 Manual Deploy 클릭**
  (공개 repo 연결이라 푸시 자동배포가 없음)

**로컬 개발** (앱 인트로 톱니바퀴에서 PC LAN IP로 변경):
```powershell
cd C:\coding\billiard\server
$env:PYTHONIOENCODING='utf-8'
$env:SAVE_UPLOADS='1'   # 업로드 사진 저장 (디버깅용, 로컬에서만)
.venv\Scripts\python.exe -m uvicorn main:app --host 0.0.0.0 --port 8000
```

임시 외부 공개가 필요하면 `start_server.bat` (Cloudflare 터널, URL 유동적).

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
