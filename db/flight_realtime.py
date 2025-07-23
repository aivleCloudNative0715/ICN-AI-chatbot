import requests
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from datetime import datetime, timedelta, timezone

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수에서 MongoDB URI와 인천공항 API 서비스 키를 가져옵니다.
mongo_uri = os.getenv("MONGO_URI")
incheon_api_service_key = os.getenv("SERVICE_KEY")

if not mongo_uri:
    print("❌ 오류: .env 파일에서 MONGO_URI를 찾을 수 없습니다. 파일을 확인해주세요.")
    exit(1)
if not incheon_api_service_key:
    print("❌ 오류: .env 파일에서 INCHEON_API_SERVICE_KEY를 찾을 수 없습니다. 파일을 확인해주세요.")
    exit(1)

# --- MongoDB 연결 설정 ---
try:
    client = MongoClient(mongo_uri, server_api=ServerApi('1'))
    db = client["AirBot"]
    flight_realtime_collection = db["FlightRealtime"]
    airline_collection = db["Airline"]

    client.admin.command('ping')
    print("✅ MongoDB에 성공적으로 연결되었습니다!")

except Exception as e:
    print(f"❌ MongoDB 연결 오류: {e}")
    exit(1)

# --- 항공사 한글명 -> IATA 코드 매핑 딕셔너리 생성 ---
def get_airline_code_map():
    airline_map = {}
    try:
        for airline_doc in airline_collection.find({}, {'airline_name_kor': 1, 'airline_code': 1, '_id': 0}):
            if 'airline_name_kor' in airline_doc and 'airline_code' in airline_doc:
                # DB에서 로드할 때 항공사 이름의 앞뒤 공백 제거
                airline_map[airline_doc['airline_name_kor'].strip()] = airline_doc['airline_code']
        print(f"✅ Airline 컬렉션에서 {len(airline_map)}개의 항공사 코드 매핑을 로드했습니다.")
    except Exception as e:
        print(f"❌ Airline 컬렉션 로드 오류: {e}")
    return airline_map

airline_code_map = get_airline_code_map()

# --- API 데이터 가져오기 함수 ---
def fetch_flight_data(api_url, search_date, direction_korean_label, service_key):
    all_flight_items = []
    page_no = 1
    total_count = 1

    # API는 한번에 1000개까지 데이터를 제공할 수 있지만, 안정성을 위해 999로 설정
    num_of_rows = 999 

    print(f"\n--- {direction_korean_label} 항공편 데이터 수집 시작 ({search_date}) ---")

    while len(all_flight_items) < total_count:
        params = {
            'serviceKey': service_key,
            'type': 'json',                   # JSON 응답 타입 지정
            'searchday': search_date,         # 조회일자
            'from_time': '0000',              # 조회시간 시작
            'to_time': '2400',                # 조회시간 끝
            'lang': 'K',                      # 언어구분 (국문)
            'numOfRows': num_of_rows,         # 한 페이지당 결과 수
            'pageNo': page_no                 # 페이지 번호
        }
        
        try:
            response = requests.get(api_url, params=params, timeout=10)
            response.raise_for_status() # HTTP 오류가 발생하면 예외를 발생시킵니다.
            json_data = response.json()

            # API 응답 구조 확인 및 데이터 추출
            body = json_data.get('response', {}).get('body', {})
            
            # 'items' 키 아래에 바로 리스트가 있습니다.
            items = body.get('items', []) 

            # 단일 항목인 경우 API가 객체를 반환할 수 있으므로 리스트로 변환 (이 로직은 유지)
            if isinstance(items, dict):
                items = [items]
            
            total_count = body.get('totalCount', 0)
            
            if not items and total_count == 0: # items가 비어있고 totalCount도 0이면 더 이상 데이터 없음
                print(f"    페이지 {page_no}: 더 이상 데이터가 없습니다.")
                break 
            
            all_flight_items.extend(items)
            print(f"    페이지 {page_no}에서 {len(items)}개 데이터 수집. 현재까지 총 {len(all_flight_items)}건 수집.")
            page_no += 1
            
            # 수집된 아이템 수가 totalCount 이상이면 루프 종료
            if len(all_flight_items) >= total_count:
                 break
            
        except requests.exceptions.Timeout:
            print(f"❌ API 요청 시간 초과: {direction_korean_label} 데이터 - 페이지 {page_no}")
            break
        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 오류 ({direction_korean_label} 데이터 - 페이지 {page_no}): {e}")
            print(f"    응답 내용: {response.text[:200]}...")
            break
        except Exception as e:
            print(f"❌ JSON 파싱 또는 데이터 처리 오류 ({direction_korean_label} 데이터 - 페이지 {page_no}): {e}")
            print(f"    응답 내용: {response.text[:200]}...")
            break
    
    print(f"✅ {direction_korean_label} 항공편 총 {len(all_flight_items)}건의 데이터 수집 완료.")
    return all_flight_items


# --- 데이터 MongoDB에 저장 함수 ---
def save_flight_data_to_db(flight_data, direction_korean_label, airline_map):
    total_processed = 0
    inserted_count = 0
    updated_count = 0

    print(f"\n--- {direction_korean_label} 항공편 데이터 MongoDB 저장 시작 ---")

    for item in flight_data:
        # 날짜/시간 문자열을 datetime 객체로 변환
        def parse_datetime(dt_str):
            try:
                return datetime.strptime(str(dt_str), '%Y%m%d%H%M') if dt_str else None
            except ValueError:
                return None

        # 항공사 코드 매핑 (API 응답 필드명 'airline')
        api_airline_name = item.get('airline', '').strip() # API에서 가져온 항공사 이름의 앞뒤 공백 제거
        mapped_airline_code = airline_map.get(api_airline_name, None)
        
        # FlightRealtime 스키마에 맞게 데이터 가공
        flight_doc = {
            'airline_code': mapped_airline_code, # 매핑된 코드를 사용
            'airport_code': item.get('airportCode'),
            'flight_id': item.get('flightId'),
            'direction': direction_korean_label,
            'remark': item.get('remark'),
            'scheduled_datetime': parse_datetime(item.get('scheduleDateTime')),
            'estimated_datetime': parse_datetime(item.get('estimatedDateTime')),
            'terminal_id': item.get('terminalid'),
            'gate_number': item.get('gatenumber'),
            'fid': item.get('fid'),
            'updated_at': datetime.now(timezone.utc)
        }

        # 방향에 따른 특정 필드 처리
        if direction_korean_label == '도착':
            flight_doc['carousel_number'] = item.get('carousel')
            flight_doc['exit_number'] = item.get('exitnumber')
            flight_doc['chkin_range'] = None
        elif direction_korean_label == '출발':
            flight_doc['chkin_range'] = item.get('chkinrange')
            flight_doc['carousel_number'] = None
            flight_doc['exit_number'] = None
        
        # FID는 필수값이므로 없으면 건너뜀 (고유 식별자로 사용)
        if not flight_doc.get('fid'):
            print(f"⚠️ 경고: FID가 없는 레코드를 건너뜠습니다. 편명: {item.get('flightId')}")
            continue

        # 항공사 코드를 찾지 못했으면 경고 출력 후 건너뛰기
        if not mapped_airline_code:
            print(f"⚠️ 경고 ({direction_korean_label}편): 항공사명 {repr(api_airline_name)}에 대한 코드를 'Airline' 컬렉션에서 찾을 수 없습니다. 이 항목은 MongoDB에 삽입되지 않고 건너뜁니다.")
            continue

        # MongoDB에 upsert (업데이트 또는 삽입)
        try:
            result = flight_realtime_collection.update_one(
                {'fid': flight_doc['fid']},
                {'$set': flight_doc},
                upsert=True
            )
            total_processed += 1
            if result.upserted_id:
                inserted_count += 1
            elif result.modified_count > 0:
                updated_count += 1

        except Exception as e:
            print(f"❌ MongoDB 저장 오류 (FID: {flight_doc.get('fid')}): {e}")

    print(f"\n--- {direction_korean_label} 항공편 데이터 MongoDB 저장 완료 ---")
    print(f"총 {total_processed}건의 데이터 처리.")
    print(f"새로 {inserted_count}건 삽입, {updated_count}건 업데이트.")


# --- 메인 실행 ---
if __name__ == "__main__":
    kst_now = datetime.now(timezone(timedelta(hours=9)))
    today_str = kst_now.strftime('%Y%m%d') 

    arrival_api_url = "http://apis.data.go.kr/B551177/StatusOfPassengerFlightsDeOdp/getPassengerArrivalsDeOdp"
    arrival_flights = fetch_flight_data(arrival_api_url, today_str, "도착", incheon_api_service_key)
    if arrival_flights:
        save_flight_data_to_db(arrival_flights, "도착", airline_code_map)

    departure_api_url = "http://apis.data.go.kr/B551177/StatusOfPassengerFlightsDeOdp/getPassengerDeparturesDeOdp"
    departure_flights = fetch_flight_data(departure_api_url, today_str, "출발", incheon_api_service_key)
    if departure_flights:
        save_flight_data_to_db(departure_flights, "출발", airline_code_map)

    client.close()
    print("\n--- MongoDB 연결 종료 ---")