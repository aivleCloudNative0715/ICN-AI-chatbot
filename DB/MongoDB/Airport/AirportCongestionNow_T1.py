from pymongo import MongoClient
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import os
from ..Key.key_manager import get_valid_api_key

load_dotenv()  # .env 파일에서 환경변수 불러오기
mongo_uri = os.getenv("MONGO_URI")

# MongoDB 설정
client = MongoClient(mongo_uri)
db = client["AirBot"]
collection = db["AirportCongestionNow_T1"]

def fetch_and_save_to_mongodb():
    url = 'http://apis.data.go.kr/B551177/StatusOfArrivals/getArrivalsCongestion'
    params_base = {
        'numOfRows': '1000',
        'pageNo': '1',
        'terno': 'T1',
        'type': 'json'
    }

    # type='public' 키 요청
    service_key = get_valid_api_key(url, params_base, key_type="public", auth_param_name="serviceKey")

    if not service_key:
        print("유효한 API 키를 찾지 못해 작업을 종료합니다.")
        return

    params = params_base.copy()
    params['serviceKey'] = service_key

    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time_str}] API 요청 시작...")

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        items = data.get("response", {}).get("body", {}).get("items", [])
        if not items:
            print(f"[{current_time_str}] 'items'가 비어 있습니다.")
            return

        # 기존 문서 전체 삭제
        collection.delete_many({})
        print(f"[{current_time_str}] 기존 문서 삭제 완료.")

        # 새로운 문서 목록 생성
        docs = []
        for item in items:
            flight_id = item.get('flightid', '').strip()
            estimated_time = item.get('estimatedtime', '').strip()
            congestion_now_id = f"{flight_id}_{estimated_time}"

            doc = {
                "congestion_now_id": congestion_now_id,
                "entry_gate": item.get('entrygate', '').strip(),
                "estimated_time": estimated_time,
                "gate_number": item.get('gatenumber', '').strip(),
                "korean": float(item.get('korean', '0')),
                "foreigner": float(item.get('foreigner', '0')),
                "scheduled_time": item.get('scheduletime', '').strip(),
                "terminal_number": item.get('terno', '').strip()
            }
            docs.append(doc)

        # 문서 일괄 삽입
        collection.insert_many(docs)
        print(f"[{current_time_str}] MongoDB 새 데이터 삽입 완료. 총 {len(docs)}개 문서.")

    except Exception as e:
        print(f"[{current_time_str}] 오류 발생: {e}")


# --- 주기적 실행 ---
if __name__ == "__main__":
    interval_seconds = 60 * 60 * 2  # 2시간

    while True:
        fetch_and_save_to_mongodb()
        print(f"\n다음 업데이트까지 {interval_seconds // 60}분 대기...\n")
        time.sleep(interval_seconds)