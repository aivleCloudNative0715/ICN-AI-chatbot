import requests
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
import re
from dotenv import load_dotenv
from ..Key.key_manager import get_valid_api_key

load_dotenv()

def update_parking_walk_time():
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

        temp_collection_name = "ParkingLotWalkTime_temp"
        main_collection_name = "ParkingLotWalkTime"

        temp_col = db[temp_collection_name]
        main_col = db[main_collection_name]

        # 임시 콜렉션 초기화 (존재하면 삭제)
        if temp_collection_name in db.list_collection_names():
            temp_col.drop()
            print(f"기존 임시 컬렉션 '{temp_collection_name}' 삭제 완료.")

        params = {
            "page": 1,
            "perPage": 100,
            "returnType": "JSON"
        }

        # 유효한 API 키 요청
        service_key = get_valid_api_key(api_url, params, key_type="public", auth_param_name="serviceKey")
        if not service_key:
            print("유효한 API 키를 찾지 못해 작업을 종료합니다.")
            return

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
            # 소요시간(분) 필드가 '00분00초' 형식이면 초와 분 합산
            duration_seconds = sum(int(val) * (60 if unit == '분' else 1) for val, unit in re.findall(r'(\d+)(분|초)', row.get("소요시간(분)", '00분00초')))

            parking_type = None
            floor = None

            if "단기주차장" in parking_type_floor:
                parking_type = "단기주차장"
            elif "장기주차장" in parking_type_floor:
                parking_type = "장기주차장"
            elif "예약주차장" in parking_type_floor:
                parking_type = "예약주차장"

            floor = parking_type_floor.replace(parking_type or '', '').strip() or None

            query = {
                "parking_type": parking_type,
                "floor": floor,
                "zone": zone,
                "terminal": terminal
            }

            parking_lot_doc = parking_lot_col.find_one(query)

            if parking_lot_doc:
                # parking_lot_doc 필드 안전 추출 및 조합
                parking_type_doc = parking_lot_doc.get('parking_type', '')
                floor_doc = parking_lot_doc.get('floor')
                zone_doc = parking_lot_doc.get('zone', '')
                terminal_doc = parking_lot_doc.get('terminal', '')

                parts = []
                if terminal_doc:
                    parts.append(f"{terminal_doc} 터미널")
                if parking_type_doc:
                    parts.append(f"{parking_type_doc}")
                if floor_doc is not None and str(floor_doc).strip():
                    parts.append(f"{floor_doc}")
                if zone_doc:
                    parts.append(f"{zone_doc} 구역")

                parking_lot_id_str = " ".join(parts)

                doc = {
                    "parkingLot_id": parking_lot_id_str,
                    "check_in_counter": checkin_counter,
                    "duration_minutes": duration_seconds
                }
                inserted_docs.append(doc)
            else:
                print(f"⚠️ 일치하는 ParkingLot 없음: {query}")

        if inserted_docs:
            # 임시 콜렉션에 한꺼번에 삽입
            temp_col.insert_many(inserted_docs)
            print(f"임시 컬렉션 '{temp_collection_name}'에 {len(inserted_docs)}개 문서 삽입 완료.")

            # 기존 콜렉션 삭제
            if main_collection_name in db.list_collection_names():
                main_col.drop()
                print(f"기존 컬렉션 '{main_collection_name}' 삭제 완료.")

            # 임시 컬렉션 이름 변경하여 교체
            db[temp_collection_name].rename(main_collection_name)
            print(f"임시 컬렉션을 '{main_collection_name}'으로 이름 변경 완료.")
        else:
            print("삽입할 문서가 없습니다. 기존 데이터 유지합니다.")

    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        client.close()
        print("MongoDB 연결이 닫혔습니다.")
