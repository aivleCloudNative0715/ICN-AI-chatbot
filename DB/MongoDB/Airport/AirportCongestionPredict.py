import requests
from datetime import datetime
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from ..Key.key_manager import get_valid_api_key
from zoneinfo import ZoneInfo

load_dotenv()

def fetch_and_save_airport_congestion_predict():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("오류: .env 파일에서 MONGO_URI를 찾을 수 없습니다. 파일을 확인해주세요.")

    client = None
    try:
        client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        db = client["AirBot"]
        collection_name = "AirportCongestionPredict"
        temp_collection_name = collection_name + "_temp"
        collection_temp = db[temp_collection_name]

        url = 'http://apis.data.go.kr/B551177/PassengerNoticeKR/getfPassengerNoticeIKR'
        params_base = {
            'selectdate': '0',
            'type': 'json'
        }

        service_key = get_valid_api_key(url, params_base, key_type="public", auth_param_name="serviceKey")
        if not service_key:
            print("유효한 API 키를 찾지 못해 작업을 종료합니다.")
            return

        params = params_base.copy()
        params['serviceKey'] = service_key

        current_time_str = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time_str}] API 요청 시작...")

        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()

        items = data.get("response", {}).get("body", {}).get("items", [])
        if not items:
            print(f"[{current_time_str}] 'items'가 비어 있습니다.")
            return

        docs = []
        for item in items:
            date = item.get('adate', '').strip()
            time_range = item.get('atime', '').strip()

            if date == '합계' and time_range == '합계':
                congestion_predict_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime('%Y%m%d')
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
            docs.append(doc)

        # 임시 컬렉션 초기화 후 한 번에 삽입
        collection_temp.delete_many({})
        if docs:
            collection_temp.insert_many(docs)

        # 기존 컬렉션 삭제 후 임시 컬렉션 이름 변경
        if collection_name in db.list_collection_names():
            db.drop_collection(collection_name)
        db[temp_collection_name].rename(collection_name)

        print(f"[{current_time_str}] MongoDB 저장 완료. 총 {len(docs)}개 문서.")

    except Exception as e:
        current_time_str = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time_str}] 오류 발생: {e}")

    finally:
        if client:
            client.close()


if __name__ == "__main__":
    fetch_and_save_airport_congestion_predict()
