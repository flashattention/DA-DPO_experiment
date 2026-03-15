import os
import time
import sys
import requests
from dotenv import load_dotenv  # 👈 추가

# 🚨 .env 파일을 현재 경로에서 로드합니다.
load_dotenv()

# 🚨 여기에 발급받은 슬랙 웹훅 URL을 입력하세요!
SLACK_WEBHOOK_URL = os.environ.get("SLACK_WEBHOOK_URL")

def send_slack_message(message):
    payload = {"text": message}
    try:
        response = requests.post(SLACK_WEBHOOK_URL, json=payload)
        response.raise_for_status()
        print("[System] 슬랙 알림 전송 완료!")
    except Exception as e:
        print(f"[Error] 슬랙 메시지 전송 실패: {e}")

def is_process_running(pid):
    """리눅스 시스템에서 해당 PID가 실행 중인지 확인합니다."""
    try:
        os.kill(pid, 0) # 프로세스에 시그널 0을 보내서 응답 확인
        return True
    except ProcessLookupError:
        return False
    except PermissionError:
        # 권한이 없어도 프로세스가 존재한다는 뜻이므로 True
        return True

def main():
    if len(sys.argv) < 3:
        print("사용법: python slack_notifier.py <감시할_PID> <로그파일경로>")
        sys.exit(1)

    target_pid = int(sys.argv[1])
    log_file = sys.argv[2]

    print(f"👀 모니터링 시작: PID {target_pid} / 로그 파일: {log_file}")
    send_slack_message(f"🚀 *[학습 모니터링 시작]*\n서버(`sv3-a100`)에서 PID `{target_pid}`(DPO 학습 7종)의 감시를 시작합니다.")

    # 1분(60초)마다 프로세스가 살아있는지 체크
    while is_process_running(target_pid):
        time.sleep(60)

    # 루프를 빠져나왔다는 것은 프로세스가 종료되었음을 의미함
    print(f"✅ PID {target_pid} 프로세스 종료 감지됨!")

    # 로그의 마지막 10줄을 읽어서 에러로 죽었는지, 정상 종료되었는지 파악
    last_logs = ""
    try:
        last_logs = os.popen(f"tail -n 10 {log_file}").read()
    except Exception:
        last_logs = "로그 파일을 읽을 수 없습니다."

    # 완료 메시지 포맷팅
    finish_message = (
        f"🎯 *[학습 종료 알림]*\n"
        f"감시 중이던 PID `{target_pid}` 의 작업이 종료되었습니다!\n\n"
        f"📝 *최근 로그 10줄:*\n```\n{last_logs}\n```"
    )
    
    send_slack_message(finish_message)

if __name__ == "__main__":
    main()