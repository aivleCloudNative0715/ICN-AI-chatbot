import requests
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
from ..Key.key_manager import get_valid_api_key

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
api_url = "https://api.odcloud.kr/api/15063436/v1/uddi:61eb754a-4644-4ab0-b12b-94310777a12e"

if not MONGO_URI:
    raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다.")


client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("MongoDB Atlas에 성공적으로 연결되었습니다.")

    db = client['AirBot']
    parking_lot_col = db['ParkingLot']
    walking_time_col = db['ParkingLotWalkTime']

    params = {
        "page": 1,
        "perPage": 100,
        "returnType": "JSON"
    }
    
    # type='public' 키 요청
    service_key = get_valid_api_key(api_url, params, key_type="public", auth_param_name="serviceKey")

    if not service_key:
        print("유효한 API 키를 찾지 못해 작업을 종료합니다.")
        exit

    params = params.copy()
    params["serviceKey"] = service_key 
    

    print("API 호출 중...")
    response = requests.get(api_url, params=params)
    response.raise_for_status()
    data = response.json()

    rows = data.get("data", [])
    print(f"{len(rows)}개의 항목을 받아왔습니다.")

    inserted_docs = []

    for row in rows:
        zone = row.get("구역")
        parking_type_floor = str(row.get("주차장", "")).strip()
        terminal = row.get("터미널")
        checkin_counter = row.get("체크인카운터")
        duration_seconds = int(row.get("소요시간(분)", 0))

        parking_type = None
        floor = None

        # parking_type 추출
        if "단기주차장" in parking_type_floor:
            parking_type = "단기주차장"
        elif "장기주차장" in parking_type_floor:
            parking_type = "장기주차장"
        elif "예약주차장" in parking_type_floor:
            parking_type = "예약주차장"

        # floor 추출
        floor = parking_type_floor.replace(parking_type or '', '').strip()

        # MongoDB에서 ParkingLot 조회
        query = {
            "parking_type": parking_type,
            "floor": floor if floor else None,
            "zone": zone,
            "terminal": terminal
        }

        parking_lot_doc = parking_lot_col.find_one(query)

        if parking_lot_doc:
            doc = {
                "parkingLot_id": parking_lot_doc["_id"],
                "check_in_counter": checkin_counter,
                "duration_minutes": duration_seconds
            }
            inserted_docs.append(doc)
        else:
            print(f"⚠️ 일치하는 ParkingLot 없음: {query}")

    # 4. MongoDB에 삽입
    if inserted_docs:
        result = walking_time_col.insert_many(inserted_docs)
        print(f"{len(result.inserted_ids)}개의 문서가 ParkingWalkingTime 컬렉션에 삽입되었습니다.")
    else:
        print("삽입할 문서가 없습니다.")

except Exception as e:
    print(f"오류 발생: {e}")
finally:
    client.close()
    print("MongoDB 연결이 닫혔습니다.")