import requests
import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import json
from datetime import datetime

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수에서 MongoDB URI와 API 서비스 키를 가져옵니다.
mongo_uri = os.getenv("MONGO_URI")
service_key = os.getenv("SERVICE_KEY")

if not mongo_uri:
    print("❌ 오류: .env 파일에서 MONGO_URI를 찾을 수 없습니다. 파일을 확인해주세요.")
    exit(1)
if not service_key:
    print("❌ 오류: .env 파일에서 SERVICE_KEY를 찾을 수 없습니다. API 키를 추가해주세요.")
    exit(1)

# --- MongoDB 연결 설정 ---
try:
    client = MongoClient(mongo_uri, server_api=ServerApi('1'))
    db = client["AirBot"]

    client.admin.command('ping')
    print("✅ MongoDB에 성공적으로 연결되었습니다!")

except Exception as e:
    print(f"❌ MongoDB 연결 오류: {e}")
    exit(1)

# --- API 엔드포인트 설정 ---
FLIGHT_SCHEDULE_ARRIVAL_API_URL = "http://apis.data.go.kr/B551177/PaxFltSched/getPaxFltSchedArrivals"
FLIGHT_SCHEDULE_DEPARTURE_API_URL = "http://apis.data.go.kr/B551177/PaxFltSched/getPaxFltSchedDepartures" # 출발 편 API URL 추가

# --- 항공사명-코드 매핑 딕셔너리를 로드하는 함수 ---
def load_airline_mapping(db):
    """
    'Airline' 컬렉션에서 항공사 한글명과 IATA 코드를 가져와 매핑 딕셔너리를 생성합니다.
    """
    print("\n--- 항공사명-코드 매핑 정보 로드 중 ---")
    airline_collection = db["Airline"]
    airlines = airline_collection.find({}, {'_id': 0, 'airline_name_kor': 1, 'airline_code': 1})
    
    airline_map = {}
    for airline in airlines:
        name = airline.get('airline_name_kor')
        code = airline.get('airline_code')
        if name and code:
            airline_map[name.strip()] = code.strip() # 공백 제거하여 매핑
    
    print(f"✅ 항공사 매핑 정보 로드 완료. 총 {len(airline_map)}건.")
    return airline_map

# --- 공통 API 데이터 업로드 함수 (도착/출발 공용) ---
def fetch_and_upload_flight_schedule_data(db, service_key, airline_map, api_url, direction_kr, missing_airlines_set):
    """
    여객편 정기편 운항 일정 API를 호출하여 데이터를 가져와 'FlightSchedule' 컬렉션에 삽입합니다.
    (도착/출발 공용 함수)
    매핑되지 않은 항공사 이름을 missing_airlines_set에 추가하고, 해당 스케줄 항목은 건너뜁니다.
    """
    collection_name = "FlightSchedule"
    all_extracted_data = []
    page_no = 1
    total_count = -1
    num_of_rows = 100 # 한 페이지당 가져올 데이터 수

    print(f"\n--- '{direction_kr}'편 스케줄 데이터 업로드 시작 ---")

    while True:
        try:
            params = {
                'serviceKey': service_key,
                'type': 'json',
                'numOfRows': num_of_rows,
                'pageNo': page_no,
                'lang': 'K' # 국문으로 설정
            }
            
            print(f"API 요청 중: {api_url} (페이지: {page_no}, {num_of_rows}개씩)")
            response = requests.get(api_url, params=params)
            response.raise_for_status()
            
            json_data = response.json()

            result_code = json_data.get('response', {}).get('header', {}).get('resultCode')
            result_msg = json_data.get('response', {}).get('header', {}).get('resultMsg')

            if result_code != '00':
                print(f"❌ API 응답 오류: [Result Code: {result_code}, Result Message: {result_msg}]")
                break

            body = json_data.get('response', {}).get('body', {})
            items = body.get('items', [])
            
            if not isinstance(items, list) and isinstance(items, dict) and 'item' in items:
                items = items['item']
                if not isinstance(items, list):
                    items = [items]
            elif not isinstance(items, list):
                 items = []

            current_page_data = []
            for item in items:
                airline_name_from_api = item.get('airline')
                cleaned_airline_name = str(airline_name_from_api).strip() if airline_name_from_api else None
                
                airline_code = airline_map.get(cleaned_airline_name, None)
                
                if airline_code is None:
                    print(f"⚠️ 경고 ({direction_kr}편): 항공사명 '{cleaned_airline_name}'에 대한 코드를 'Airline' 컬렉션에서 찾을 수 없습니다. 이 항목은 MongoDB에 삽입되지 않고 건너뜁니다.")
                    missing_airlines_set.add(cleaned_airline_name) # 매핑되지 않은 항공사 이름을 집합에 추가
                    continue # 이 항목은 건너뛰고 다음 항목으로 넘어갑니다.
                    
                st_str = str(item.get('st', '')).zfill(4)
                scheduled_time = f"{st_str[:2]}:{st_str[2:]}" if len(st_str) == 4 else None

                first_date_str = str(item.get('firstdate', ''))
                first_date = datetime.strptime(first_date_str, '%Y%m%d') if first_date_str else None 
                
                last_date_str = str(item.get('lastdate', ''))
                last_date = datetime.strptime(last_date_str, '%Y%m%d') if last_date_str else None

                monday = item.get('monday', 'N').upper() == 'Y'
                tuesday = item.get('tuesday', 'N').upper() == 'Y'
                wednesday = item.get('wednesday', 'N').upper() == 'Y'
                thursday = item.get('thursday', 'N').upper() == 'Y'
                friday = item.get('friday', 'N').upper() == 'Y'
                saturday = item.get('saturday', 'N').upper() == 'Y'
                sunday = item.get('sunday', 'N').upper() == 'Y'

                current_page_data.append({
                    'airline_name_kor': airline_name_from_api,
                    'airline_code': airline_code, # 여기에는 이제 반드시 매핑된 코드가 들어갑니다.
                    'airport_code': item.get('airportcode'),
                    'scheduled_time': scheduled_time,
                    'first_date': first_date,
                    'last_date': last_date,
                    'direction': direction_kr,
                    'season': item.get('season'),
                    'monday': monday,
                    'tuesday': tuesday,
                    'wednesday': wednesday,
                    'thursday': thursday,
                    'friday': friday,
                    'saturday': saturday,
                    'sunday': sunday
                })
            
            if not current_page_data:
                print(f"⚠️ 페이지 {page_no}에 추출되어 삽입할 데이터가 없습니다. (더 이상 매핑 가능한 데이터 없음 또는 오류)")
                break

            all_extracted_data.extend(current_page_data)
            
            total_count = body.get('totalCount', 0)
            print(f"현재까지 {len(all_extracted_data)}건 수집 (총 {total_count}건 예상). 다음 페이지 준비 중...")

            # API의 totalCount는 매핑 실패 항목을 포함한 총 건수이므로,
            # 실제 삽입될 데이터 건수와 다를 수 있습니다.
            # 하지만 모든 페이지를 순회하기 위해 totalCount를 기준으로 계속 진행합니다.
            # 만약 현재 수집된 데이터가 totalCount보다 많거나 같으면, 더 이상 가져올 데이터가 없다고 판단합니다.
            if len(all_extracted_data) >= total_count and len(items) < num_of_rows:
                 # items 길이가 num_of_rows보다 작으면 마지막 페이지로 간주
                 break
            
            page_no += 1

        except requests.exceptions.RequestException as e:
            print(f"❌ API 요청 중 오류가 발생했습니다: {e}")
            break
        except Exception as e:
            print(f"❌ 데이터 처리 중 오류가 발생했습니다: {e}")
            if 'json_data' in locals():
                print(f"디버깅용 전체 응답: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
            break

    if all_extracted_data:
        df = pd.DataFrame(all_extracted_data)

        # 데이터프레임의 NaN/빈 문자열을 None으로 변환 (MongoDB 삽입 시 BSON Null로 처리)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: None if (pd.isna(x) or (isinstance(x, str) and x.strip() == '')) else x.strip() if isinstance(x, str) else x)

        data_to_insert = df.to_dict(orient="records")

        collection = db[collection_name]
        
        # **주의: 기존 데이터를 삭제하고 새로 넣고 싶을 때만 아래 줄의 주석을 해제하세요.**
        # collection.delete_many({'direction': direction_kr}) # 특정 방향의 기존 데이터 삭제 (중복 방지용)

        if data_to_insert: # 삽입할 데이터가 있을 경우에만 insert_many 호출
            collection.insert_many(data_to_insert)
            print(f"✅ 성공적으로 API 데이터를 '{collection_name}' 컬렉션에 삽입했습니다. 총 {len(data_to_insert)}건. (방향: {direction_kr})")
        else:
            print(f"⚠️ '{collection_name}' 컬렉션에 삽입할 매핑 가능한 데이터가 없습니다. (방향: {direction_kr})")
    else:
        print(f"⚠️ '{collection_name}' 컬렉션에 삽입할 매핑 가능한 데이터가 없습니다. (방향: {direction_kr})")


# --- 메인 실행 부분 ---

print("\n--- 여객편 정기편 운항 일정 정보 업로드 시작 ---")

# 1. 항공사 매핑 정보 먼저 로드 (도착/출발 모두에 사용)
airline_mapping_dict = load_airline_mapping(db)
if not airline_mapping_dict:
    print("❌ 오류: 항공사 매핑 정보를 로드할 수 없습니다. 'Airline' 컬렉션을 확인해주세요.")
    client.close()
    exit(1)

# 매핑되지 않은 항공사 이름을 저장할 집합 (전체 스크립트 실행 동안 누적)
missing_airlines_overall = set()

# 2. 도착 편 스케줄 데이터 업로드
fetch_and_upload_flight_schedule_data(db, service_key, airline_mapping_dict, FLIGHT_SCHEDULE_ARRIVAL_API_URL, '도착', missing_airlines_overall)

# 3. 출발 편 스케줄 데이터 업로드
fetch_and_upload_flight_schedule_data(db, service_key, airline_mapping_dict, FLIGHT_SCHEDULE_DEPARTURE_API_URL, '출발', missing_airlines_overall)

# --- 매핑되지 않은 항공사 목록을 파일로 저장 ---
if missing_airlines_overall:
    sorted_missing_airlines = sorted(list(missing_airlines_overall))
    
    # 출력 폴더 생성 (없으면)
    output_directory = "./db/output"
    os.makedirs(output_directory, exist_ok=True)
    
    output_csv_path = os.path.join(output_directory, "unmatched_airlines_for_schedules.csv")
    
    df_missing = pd.DataFrame(sorted_missing_airlines, columns=['airline_name_from_schedule_api'])
    df_missing.to_csv(output_csv_path, index=False, encoding='utf-8-sig')
    
    print(f"\n✅ 매핑되지 않은 항공사 목록을 '{output_csv_path}' 파일로 저장했습니다. 총 {len(sorted_missing_airlines)}개.")
    print("파일을 열어 확인하고, 필요시 'Airline' 컬렉션에 해당 항공사를 추가하거나, 데이터를 정제하세요.")
else:
    print("\n✅ API 스케줄에서 'Airline' 컬렉션에 매핑되지 않은 항공사를 찾지 못했습니다. (모든 항공사가 성공적으로 매핑됨)")

# --- MongoDB 연결 종료 ---
client.close()
print("\n--- 데이터 업로드 완료 및 MongoDB 연결 종료 ---")