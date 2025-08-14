import requests
import os
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

def update_parking_walk_time():
    """
    공항 주차장 소요시간 데이터를 API에서 가져와
    ParkingLot 컬렉션과 매칭 후
    ParkingLotWalkTime 컬렉션에 저장하는 함수
    """

    service_key = os.getenv("PARKING_WALK_TIME_API_KEY") or \
                  "BKp1kHZdj/1XpNErxqOFVQPHeiZmmMDhLH/3SBqhOpEGqaD1AeTVPUndV81fQnoNNuAACLI32ySPHmJCV8DGTQ=="
    MONGO_URI = os.getenv("MONGO_URI") or \
                "mongodb+srv://ninguis555:xAog5CN4Mgt4sl05@aivle0715.quxcjjc.mongodb.net/?retryWrites=true&w=majority"
    api_url = os.getenv("PARKING_WALK_TIME_API_URL") or \
              "https://api.odcloud.kr/api/15063436/v1/uddi:61eb754a-4644-4ab0-b12b-94310777a12e"

    if not service_key:
        raise ValueError("PARKING_WALK_TIME_API_KEY 환경 변수가 설정되지 않았습니다.")
    if not MONGO_URI:
        raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다.")
    if not api_url:
        raise ValueError("PARKING_WALK_TIME_API_URL 환경 변수가 설정되지 않았습니다.")

    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

    try:
        client.admin.command('ping')
        print("✅ MongoDB Atlas에 연결 성공")

        db = client['AirBot']
        parking_lot_col = db['ParkingLot']
        walking_time_col = db['ParkingLotWalkTime']

        params = {
            "serviceKey": service_key,
            "page": 1,
            "perPage": 100,
            "returnType": "JSON"
        }

        print("🌐 API 호출 중...")
        response = requests.get(api_url, params=params)
        response.raise_for_status()
        data = response.json()

        rows = data.get("data", [])
        print(f"📦 {len(rows)}개의 데이터 수신")

        inserted_docs = []

        for row in rows:
            zone = row.get("구역")
            parking_type_floor = str(row.get("주차장", "")).strip()
            terminal = row.get("터미널")
            checkin_counter = row.get("체크인카운터")

            # 소요시간 변환 (MM:SS -> 초)
            duration_raw = str(row.get("소요시간(분)", "0"))
            if ":" in duration_raw:
                minutes, seconds = map(int, duration_raw.split(":"))
                duration_seconds = minutes * 60 + seconds
            elif duration_raw.isdigit():
                duration_seconds = int(duration_raw) * 60
            else:
                import re
                duration_seconds = sum(
                    int(val) * (60 if unit == '분' else 1)
                    for val, unit in re.findall(r'(\d+)(분|초)', duration_raw)
                )

            # parking_type, floor 추출
            types = ["단기주차장", "장기주차장", "예약주차장"]
            levels = ["지상", "지하"]

            parking_type = next((t for t in types if parking_type_floor.startswith(t)), "")
            rest = parking_type_floor[len(parking_type):].strip()
            level = next((lv for lv in levels if rest.startswith(lv)), "")

            # 잘린 경우 보정
            if not level:
                if rest.startswith("상"):
                    level = "지상"
                    rest = "지상" + rest[1:]
                elif rest.startswith("하"):
                    level = "지하"
                    rest = "지하" + rest[1:]

            floor_number = rest[len(level):].strip()
            floor = f"{level}{floor_number}" if level else rest

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
                    "duration_seconds": duration_seconds
                }
                inserted_docs.append(doc)
            else:
                print(f"⚠️ 일치하는 ParkingLot 없음: {query}")

        # MongoDB 삽입
        if inserted_docs:
            result = walking_time_col.insert_many(inserted_docs)
            print(f"📥 {len(result.inserted_ids)}개 문서 삽입 완료")
        else:
            print("ℹ️ 삽입할 문서 없음")

    except Exception as e:
        print(f"❌ 오류 발생: {e}")
    finally:
        client.close()
        print("🔌 MongoDB 연결 종료")


# 외부에서 바로 실행 시에도 동작
if __name__ == "__main__":
    update_parking_walk_time()
