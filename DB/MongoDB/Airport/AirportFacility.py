from pymongo import MongoClient
import requests
import json
import time
from datetime import datetime
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta
from dateutil.relativedelta import relativedelta
from key_manager import get_valid_api_key

load_dotenv()  # .env 파일에서 환경변수 불러오기
mongo_uri = os.getenv("MONGO_URI")

# MongoDB 설정
client = MongoClient(mongo_uri)
db = client["AirBot"]
collection = db["AirportFacility"]

def fetch_and_save_to_mongodb():
    url = 'http://apis.data.go.kr/B551177/FacilitiesInformation/getFacilitesInfo'
    params_base ={ 
            'numOfRows' : '10000', 
            'pageNo' : '1',
            'lang' : 'K', 
            'type' : 'json' }

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
        
        saved_count = 0

        for item in items:
            facility_id = str(item.get('sn') or '').strip()

            doc = {
                "facility_id": facility_id,
                "arrordep": (item.get('arrordep') or '').strip(),
                "item": (item.get('facilityitem') or '').strip(),
                "facility_name": (item.get('facilitynm') or '').strip(),
                "location": (item.get('lcnm') or '').strip(),
                "large_category": (item.get('lcategorynm') or '').strip(),
                "medium_category": (item.get('mcategorynm') or '').strip(),
                "small_category": (item.get('scategorynm') or '').strip(),
                "service_time": (item.get('servicetime') or '').strip(),
                "is_duty_free_location": (item.get('lcduty') or '').strip(),
                "terminal_id": (item.get('terminalid') or '').strip(),
                "floor_info": (item.get('floorinfo') or '').strip(),
                "tel": (item.get('tel') or '').strip()
            }

            result = collection.update_one(
                {"facility_id": facility_id},
                {"$set": doc},
                upsert=True
            )

            if result.modified_count > 0 or result.upserted_id is not None:
                saved_count += 1

        print(f"[{current_time_str}] MongoDB 저장 완료. 총 {saved_count}개 문서.")

    except Exception as e:
        print(f"[{current_time_str}] 오류 발생: {e}")

# 이건 한달마다 한번만 실행
if __name__ == "__main__":
    fetch_and_save_to_mongodb()

    now = datetime.now()
    next_run = now + relativedelta(day=2)

    print(f"실행 완료. 현재 시각: {now.strftime('%Y-%m-%d %H:%M')}")
    print(f"다음 실행 권장 시각: {next_run.strftime('%Y-%m-%d %H:%M')}")
# API에서 중복된 데이터가 있어서 총 저장 갯수는 914개