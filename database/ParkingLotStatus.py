import requests
import re
import os
import time
import schedule
from datetime import datetime
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

# 전처리용 파싱 함수
def parse_floor(floor_text):
    terminal_match = re.match(r'(T\d)\s*(.+)', floor_text)
    if terminal_match:
        terminal = terminal_match.group(1)
        rest = terminal_match.group(2)
    else:
        terminal = ''
        rest = floor_text.strip()

    floor_match = re.search(r'(지하M층|지하\d층|지상\d층|지상층|지하층|타워|타워층)$', rest)
    floor = floor_match.group(0) if floor_match else ''
    parking_raw = rest.replace(floor, '').strip()

    parking_type = '기타'
    zone = ''
    if '장기' in parking_raw:
        parking_type = '장기주차장'
        zone_match = re.search(r'P\d', parking_raw)
        if zone_match:
            zone = zone_match.group(0)
    elif '단기' in parking_raw:
        parking_type = '단기주차장'
    elif '예약' in parking_raw:
        parking_type = '예약주차장'
    else:
        parking_type = parking_raw

    return terminal, parking_type, zone, floor

load_dotenv()

service_key = os.getenv('PARKING_LOT_STATUS_API_KEY')
MONGO_URI = os.getenv('MONGO_URI')
parking_lot_status_url = os.getenv("PARKING_LOT_STATUS_API_URL")

if not service_key:
    raise ValueError("PARKING_LOT_STATUS_API_KEY 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
if not MONGO_URI:
    raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")
if not parking_lot_status_url:
    raise ValueError("PARKING_LOT_STATUS_API_URL 환경 변수가 설정되지 않았습니다.")

client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
db = client['AirBot']
parking_lot_col = db['ParkingLot']
parking_lot_status_col = db['ParkingLotStatus']

def fetch_and_insert_parking_lot_status():
    try:
        params = {
            "serviceKey": service_key,
            "numOfRows": 100,
            "pageNo": 1,
            'type': 'json'
        }
        print("[요청] 주차장 가용 정보 API 호출 중...")
        response = requests.get(parking_lot_status_url, params=params)
        response.raise_for_status()

        data = response.json()
        items = data.get("response", {}).get("body", {}).get("items", [])
        print(f"[받음] {len(items)}개의 항목 수신됨")

        inserted_docs = []

        parking_lot_status_col.delete_many({})
        print("기본 주차장 가용 정보 삭제")

        for item in items:
            floor_raw = item.get("floor", "").strip()
            parking = int(item.get("parking", 0))
            parkingarea = int(item.get("parkingarea", 0))
            datetm_str = item.get("datetm", "")[:14]
            datetm = datetime.strptime(datetm_str, "%Y%m%d%H%M%S")

            terminal, parking_type, zone, floor = parse_floor(floor_raw)

            ## MongoDB에서 ParkingLot 조회
            parking_lot_doc = parking_lot_col.find_one({
                "terminal": terminal,
                "parking_type": parking_type,
                "zone": zone if zone else None,
                "floor": floor if floor else None
            })

            if parking_lot_doc:
                doc = {
                    "parkingLot_id": parking_lot_doc["_id"],
                    "parking": parking,
                    "parking_area": parkingarea,
                    "created_at": datetm
                }
                inserted_docs.append(doc)
            else:
                print(f"⚠️ 일치하는 ParkingLot 없음: '{floor_raw}' → terminal={terminal}, parking_type={parking_type}, zone={zone}, floor={floor}")

        if inserted_docs:
            parking_lot_status_col.insert_many(inserted_docs)
            print(f"[성공] {len(inserted_docs)}개 문서 삽입 완료")
        else:
            print("[건너뜀] 삽입할 문서 없음")

    except Exception as e:
        print(f"[오류] {e}")

# 3. 스케줄 등록 (5분 간격 실행)
schedule.every(5).minutes.do(fetch_and_insert_parking_lot_status)

print("⏳ 주차장 가용 정보 스케줄러 실행 중 (5분 간격)...")

try:
    while True:
        schedule.run_pending()
        time.sleep(1)
except KeyboardInterrupt:
    print("⛔ 수동 종료됨")
finally:
    client.close()
    print("MongoDB 연결 종료")

# if __name__ == "__main__":
#     fetch_and_insert_parking_lot_status()
#     client.close()
#     print("MongoDB 연결 종료")
