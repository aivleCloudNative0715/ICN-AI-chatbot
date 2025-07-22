from pymongo import MongoClient
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일에서 환경변수 불러오기
mongo_uri = os.getenv("MONGO_URI")

# MongoDB 설정
client = MongoClient(mongo_uri)
db = client["AirBot"]
collection = db["AirportCongestionPredict"]

def fetch_and_save_to_mongodb():
    url = 'http://apis.data.go.kr/B551177/PassengerNoticeKR/getfPassengerNoticeIKR'
    params = {
        'serviceKey': '',  # 여기에 서비스키 삽입
        'selectdate': '0',
        'type': 'json'
    }

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

        for item in items:
            date = item.get('adate').strip()
            time_range = item.get('atime').strip()

            # 합계 데이터라면 현재 날짜를 congestion_predict_id로 사용
            if date == '합계' and time_range == '합계':
                congestion_predict_id = datetime.now().strftime('%Y%m%d')
            else:
                congestion_predict_id = f"{date}_{time_range}"

            doc = {
                "congestion_predict_id": congestion_predict_id,
                "date": date,
                "time": time_range,
                "t1_arrival_a_b": int(float(item.get("t1sum1", 0))),
                "t1_arrival_e_f": int(float(item.get("t1sum2", 0))),
                "t1_arrival_c": int(float(item.get("t1sum3", 0))),
                "t1_arrival__d": int(float(item.get("t1sum4", 0))),
                "t1_arrival_sum": int(float(item.get("t1sumset1", 0))),
                "t1_departure_1_2": int(float(item.get("t1sum5", 0))),
                "t1_departure_3": int(float(item.get("t1sum6", 0))),
                "t1_departure_4": int(float(item.get("t1sum7", 0))),
                "t1_departure_5_6": int(float(item.get("t1sum8", 0))),
                "t1_departure_sum": int(float(item.get("t1sumset2", 0))),
                "t2_arrival_a": int(float(item.get("t2sum1", 0))),
                "t2_arrival_b": int(float(item.get("t2sum2", 0))),
                "t2_arrival_sum": int(float(item.get("t2sumset1", 0))),
                "t2_departure_1": int(float(item.get("t2sum3", 0))),
                "t2_departure_2": int(float(item.get("t2sum4", 0))),
                "t2_departure_sum": int(float(item.get("t2sumset2", 0))),
            }

            # date + time 기준으로 중복 방지 (upsert)
            collection.update_one(
                {"congestion_predict_id": congestion_predict_id},
                {"$set": doc},
                upsert=True
            )

        print(f"[{current_time_str}] MongoDB 저장 완료. 총 {len(items)}개 문서.")

    except Exception as e:
        print(f"[{current_time_str}] 오류 발생: {e}")

# --- 주기적 실행 ---
if __name__ == "__main__":
    interval_seconds = 60 * 12  # 12시간

    while True:
        fetch_and_save_to_mongodb()
        print(f"\n다음 업데이트까지 {interval_seconds}분 대기...\n")
        time.sleep(interval_seconds)
      