import pandas as pd
import requests
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from ..Key.key_manager import get_valid_api_key
from zoneinfo import ZoneInfo

load_dotenv()

def fetch_and_save_taf_data():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("오류: .env 파일에서 MONGO_URI를 찾을 수 없습니다. 파일을 확인해주세요.")
        return

    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client["AirBot"]

        collection_name = "TAF"
        temp_collection_name = collection_name + "_temp"
        collection_temp = db[temp_collection_name]

        url = 'https://apihub.kma.go.kr/api/typ02/openApi/AmmService/getTaf'
        params_base = {
            'pageNo': '1',
            'numOfRows': '30',
            'dataType': 'JSON',
            'icao': 'RKSI'
        }

        authKey = get_valid_api_key(url, params_base, key_type="weather", auth_param_name="authKey")
        if not authKey:
            print("유효한 API 키를 찾지 못해 작업을 종료합니다.")
            return

        params = params_base.copy()
        params['authKey'] = authKey

        current_time_str = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time_str}] TAF API 요청 시작...")

        response = requests.get(url, params=params)
        response.raise_for_status()

        print(f"[{current_time_str}] Status Code: {response.status_code}")
        print(f"[{current_time_str}] API 응답을 성공적으로 받았습니다. JSON 데이터를 파싱합니다.")

        json_data = response.json()
        items = json_data.get("response", {}).get("body", {}).get("items", {}).get("item", [])

        if not items:
            print(f"[{current_time_str}] 'item'이 비어 있습니다.")
            return

        # 임시 컬렉션 초기화
        collection_temp.delete_many({})

        taf_id = datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y%m%d")

        documents = []
        for idx, item in enumerate(items):
            metar_msg = item.get("metarMsg", "").strip()
            doc = {
                "_id": f"{taf_id}_{idx}",  # 고유 키 지정 (날짜 + 인덱스)
                "taf_id": taf_id,
                "metar_MSG": metar_msg
            }
            documents.append(doc)

        if documents:
            collection_temp.insert_many(documents)

            # 기존 컬렉션 삭제 후 임시 컬렉션 rename
            if collection_name in db.list_collection_names():
                db.drop_collection(collection_name)
            collection_temp.rename(collection_name)

            print(f"[{current_time_str}] MongoDB 저장 완료. 총 {len(documents)}개 문서.")
        else:
            print(f"[{current_time_str}] 저장할 문서가 없습니다.")

    except Exception as e:
        print(f"[{current_time_str}] 오류 발생: {e}")

    finally:
        if client:
            client.close()


if __name__ == "__main__":
    fetch_and_save_taf_data()
