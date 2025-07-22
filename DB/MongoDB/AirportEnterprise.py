from pymongo import MongoClient
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import os
from datetime import datetime

load_dotenv()  # .env 파일에서 환경변수 불러오기
mongo_uri = os.getenv("MONGO_URI")

# MongoDB 설정
client = MongoClient(mongo_uri)
db = client["AirBot"]
collection = db["AirportEnterprise"]

def fetch_and_save_to_mongodb():
    url = 'http://apis.data.go.kr/B551177/StatusOfFacility/getFacilityKR'
    params ={'serviceKey' : '', 
            'type' : 'json', 
            'numOfRows' : '10000', 
            'pageNo' : '1' }

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
        saved_count = 0
        new_documents = []

        for idx, item in enumerate(items):

            doc = {
                "enterprise_id": idx + 1,
                "enterprise_name": (item.get('entrpskoreannm') or '').strip(),
                "item": (item.get('trtmntprdlstkoreannm') or '').strip(),
                "location": (item.get('lckoreannm') or '').strip(),
                "service_time": (item.get('servicetime') or '').strip(),
                "arrordep": (item.get('arrordep') or '').strip(),
                "tel": (item.get('tel') or '').strip()
            }

            new_documents.append(doc)

        if new_documents:
            collection.insert_many(new_documents)
            saved_count = len(new_documents)

        print(f"[{current_time_str}] MongoDB 저장 완료. 총 {saved_count}개 문서.")

    except Exception as e:
        print(f"[{current_time_str}] 오류 발생: {e}")

# --- 주기적 실행 ---
if __name__ == "__main__":
    interval_seconds = 60 * 60 * 24 * 2  # 2일 = 60초 * 60분 * 24시간 * 2

    while True:
        fetch_and_save_to_mongodb()
        print(f"\n다음 업데이트까지 {interval_seconds // 3600}시간 대기...\n")
        time.sleep(interval_seconds)
      
