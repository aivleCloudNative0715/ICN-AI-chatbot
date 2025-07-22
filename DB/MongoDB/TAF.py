import pandas as pd
import numpy as np
import requests
import json
import os
import time 
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv

load_dotenv()  # .env 파일에서 환경변수 불러오기
mongo_uri = os.getenv("MONGO_URI")

# MongoDB 설정
client = MongoClient(mongo_uri)
db = client["AirBot"]
collection = db["TAF"]

def fetch_and_save_taf_data():

    url = 'https://apihub.kma.go.kr/api/typ02/openApi/AmmService/getTaf'
    params = {
        'authKey': '',  # 여기에 서비스키 삽입
        'pageNo': '1',
        'numOfRows': '30',
        'dataType': 'JSON',
        'icao': 'RKSI'
    }


    current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{current_time_str}] TAF API 요청 시작...")

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()

        print(f"[{current_time_str}] Status Code: {response.status_code}")
        print(f"[{current_time_str}] API 응답을 성공적으로 받았습니다. JSON 데이터를 파싱합니다.")

        json_data = response.json()
        items = json_data.get("response", {}).get("body", {}).get("items", {}).get("item", [])

        if not items:
            print(f"[{current_time_str}] 'item'이 비어 있습니다.")
            return

        taf_id = datetime.now().strftime("%Y%m%d")

        saved_count = 0
        for item in items:
            metar_msg = item.get("metarMsg", "").strip()
            doc = {
                "taf_id": taf_id,
                "metar_MSG": metar_msg
            }

            result = collection.update_one(
                {"taf_id": taf_id},
                {"$set": doc},
                upsert=True
            )
            if result.modified_count > 0 or result.upserted_id is not None:
                saved_count += 1

        print(f"[{current_time_str}] MongoDB 저장 완료. 총 {saved_count}개 문서.")

    except Exception as e:
        print(f"[{current_time_str}] 오류 발생: {e}")

# --- 주기적 실행 ---
if __name__ == "__main__":
    interval_seconds = 60 * 60  # 1시간 마다 호출

    while True:
        fetch_and_save_taf_data()
        print(f"\n다음 업데이트까지 {interval_seconds // 60}분 대기...\n")
        time.sleep(interval_seconds)