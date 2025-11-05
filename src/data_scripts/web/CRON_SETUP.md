# Cron Job Setup Guide

이 가이드는 CoinDesk 뉴스 스크래퍼를 자동으로 실행하도록 cron job을 설정하는 방법을 설명합니다.

## 파일

- **run_daily_scraper.sh** - 일일 스크래핑을 실행하는 셸 스크립트
- 로그 파일: `logs/coindesk_scraper_YYYYMMDD.log`

## 설정 방법

### 1. 스크립트 테스트

먼저 스크립트가 제대로 작동하는지 테스트하세요:

```bash
cd /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/src/data_scripts/web
./run_daily_scraper.sh
```

로그 파일 확인:
```bash
cat /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/logs/coindesk_scraper_$(date +%Y%m%d).log
```

### 2. Crontab 설정

#### 2.1 Crontab 편집기 열기

```bash
crontab -e
```

#### 2.2 Cron Job 추가

다음 줄을 추가하세요 (시간은 원하는 대로 조정):

```bash
# 매일 오전 9시에 실행
0 9 * * * /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/src/data_scripts/web/run_daily_scraper.sh

# 또는 매일 오전 6시에 실행
0 6 * * * /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/src/data_scripts/web/run_daily_scraper.sh

# 또는 하루에 2번 (오전 9시, 오후 6시)
0 9,18 * * * /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/src/data_scripts/web/run_daily_scraper.sh

# 또는 매 6시간마다
0 */6 * * * /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/src/data_scripts/web/run_daily_scraper.sh
```

저장하고 종료 (vim의 경우: `ESC` → `:wq` → `Enter`)

#### 2.3 Cron Job 확인

```bash
crontab -l
```

### 3. Cron 시간 설정 가이드

Cron 형식: `분 시 일 월 요일 명령어`

```
* * * * * command
│ │ │ │ │
│ │ │ │ └─── 요일 (0-7, 0과 7은 일요일)
│ │ │ └───── 월 (1-12)
│ │ └─────── 일 (1-31)
│ └───────── 시 (0-23)
└─────────── 분 (0-59)
```

**예시:**
- `0 9 * * *` - 매일 오전 9시
- `30 8 * * 1-5` - 평일 오전 8시 30분
- `0 */6 * * *` - 매 6시간마다
- `0 0 * * 0` - 매주 일요일 자정

### 4. macOS에서 Cron 권한 설정

macOS Catalina 이상에서는 cron에 Full Disk Access 권한이 필요할 수 있습니다:

1. **시스템 환경설정** → **보안 및 개인 정보 보호** → **개인 정보 보호**
2. **Full Disk Access** 선택
3. 자물쇠 클릭하여 잠금 해제
4. `+` 버튼 클릭
5. `/usr/sbin/cron` 추가

또는 Terminal에 권한 부여:
1. **시스템 환경설정** → **보안 및 개인 정보 보호** → **개인 정보 보호**
2. **Full Disk Access** 선택
3. Terminal 앱 추가

### 5. 로그 확인

스크립트 실행 로그는 다음 위치에 저장됩니다:

```bash
# 오늘 로그
cat /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/logs/coindesk_scraper_$(date +%Y%m%d).log

# 모든 로그 확인
ls -lh /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/logs/

# 최근 로그 실시간 확인
tail -f /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/logs/coindesk_scraper_$(date +%Y%m%d).log
```

### 6. 데이터베이스 자동 업데이트 (선택사항)

스크래핑 후 자동으로 DuckDB에 저장하려면 `run_daily_scraper.sh` 파일을 편집하세요:

```bash
nano /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/src/data_scripts/web/run_daily_scraper.sh
```

파일 끝부분의 주석을 해제하세요:
```bash
# 이 부분의 주석 제거
echo "Ingesting to DuckDB..." >> "$LOG_FILE"
python "$PROJECT_DIR/src/database/ingest_to_duckdb.py" >> "$LOG_FILE" 2>&1
```

## 문제 해결

### Cron이 실행되지 않는 경우

1. **Cron 서비스 확인**:
   ```bash
   sudo launchctl list | grep cron
   ```

2. **시스템 로그 확인**:
   ```bash
   log show --predicate 'process == "cron"' --last 1h
   ```

3. **스크립트 권한 확인**:
   ```bash
   ls -l /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/src/data_scripts/web/run_daily_scraper.sh
   # -rwxr-xr-x 여야 함 (실행 권한)
   ```

4. **수동 실행 테스트**:
   ```bash
   /Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/src/data_scripts/web/run_daily_scraper.sh
   ```

### 가상환경을 찾을 수 없는 경우

스크립트에서 Python 경로를 절대 경로로 지정:
```bash
/Users/shawn/Documents/sjsu/2025-2/NLP/terminalC/.venv/bin/python coindesk_scraper.py
```

## Cron Job 제거

```bash
crontab -e
# 해당 줄을 삭제하고 저장
```

또는 모든 cron job 제거:
```bash
crontab -r
```

## 대안: Launchd (macOS 권장)

macOS에서는 cron 대신 launchd를 사용하는 것이 권장됩니다.
launchd 설정 파일을 만들려면 별도로 문의하세요.

## 예상 데이터 축적

- **1일**: 약 25-50개 기사
- **1주일**: 약 175-350개 기사
- **1개월**: 약 750-1,500개 기사
- **1년**: 약 9,000-18,000개 기사

RSS 피드는 최신 기사만 제공하므로, 정기적으로 실행해야 과거 데이터를 놓치지 않습니다.
