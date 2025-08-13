import requests
from datetime import datetime
from dotenv import load_dotenv
import os
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from ..Key.key_manager import get_valid_api_key
from zoneinfo import ZoneInfo

load_dotenv()

def fetch_and_save_airport_enterprise():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("오류: .env 파일에서 MONGO_URI를 찾을 수 없습니다. 파일을 확인해주세요.")

    client = None
    try:
        client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        db = client["AirBot"]
        collection_name = "AirportEnterprise"
        temp_collection_name = collection_name + "_temp"
        collection_temp = db[temp_collection_name]

        url = 'http://apis.data.go.kr/B551177/StatusOfFacility/getFacilityKR'
        params_base = {
            'type': 'json',
            'numOfRows': '10000',
            'pageNo': '1'
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

        # 임시 컬렉션 초기화
        collection_temp.delete_many({})

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
            collection_temp.insert_many(new_documents)

        # 기존 컬렉션 삭제 후 임시 컬렉션 이름 변경
        if collection_name in db.list_collection_names():
            db.drop_collection(collection_name)
        db[temp_collection_name].rename(collection_name)

        print(f"[{current_time_str}] MongoDB 저장 완료. 총 {len(new_documents)}개 문서.")

    except Exception as e:
        current_time_str = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time_str}] 오류 발생: {e}")

    finally:
        if client:
            client.close()

if __name__ == "__main__":
    fetch_and_save_airport_enterprise()
