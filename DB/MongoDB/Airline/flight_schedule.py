import requests
import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
import json
from datetime import datetime
from ..Key.key_manager import get_valid_api_key

load_dotenv()

def fetch_and_upload_flight_schedule():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("❌ .env 파일에서 MONGO_URI를 찾을 수 없습니다.")
        return

    client = MongoClient(mongo_uri, server_api=ServerApi('1'))
    db = client["AirBot"]

    collection_name = "FlightSchedule"
    tmp_collection_name = "FlightSchedule_tmp"

    airline_map = load_airline_mapping()
    print(f"항공사 매핑 정보 로드 완료. 총 {len(airline_map)}건.")
    missing_airlines_overall = set()

    api_info_list = [
        {
            "url": "http://apis.data.go.kr/B551177/PaxFltSched/getPaxFltSchedArrivals",
            "direction": "도착"
        },
        {
            "url": "http://apis.data.go.kr/B551177/PaxFltSched/getPaxFltSchedDepartures",
            "direction": "출발"
        }
    ]

    all_data = []

    for api_info in api_info_list:
        api_url = api_info["url"]
        direction_kr = api_info["direction"]
        print(f"\n--- '{direction_kr}'편 스케줄 데이터 업로드 시작 ---")

        page_no = 1
        num_of_rows = 100

        while True:
            try:
                params = {
                    'type': 'json',
                    'numOfRows': num_of_rows,
                    'pageNo': page_no,
                    'lang': 'K'
                }

                # API 키 얻는 부분, get_valid_api_key 함수는 외부에 정의돼 있다고 가정
                service_key = get_valid_api_key(api_url, params, key_type="public", auth_param_name="serviceKey")

                if not service_key:
                    print("유효한 API 키를 찾지 못해 작업을 종료합니다.")
                    return

                params['serviceKey'] = service_key

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
                        print(f"⚠️ 경고 ({direction_kr}편): 항공사명 '{cleaned_airline_name}'에 대한 코드를 찾을 수 없습니다. 건너뜁니다.")
                        missing_airlines_overall.add(cleaned_airline_name)
                        continue

                    st_str = str(item.get('st', '')).zfill(4)
                    scheduled_time = f"{st_str[:2]}:{st_str[2:]}" if len(st_str) == 4 else None

                    first_date_str = str(item.get('firstdate', ''))
                    first_date = datetime.strptime(first_date_str, '%Y%m%d') if first_date_str else None

                    last_date_str = str(item.get('lastdate', ''))
                    last_date = datetime.strptime(last_date_str, '%Y%m%d') if last_date_str else None

                    current_page_data.append({
                        'airline_name_kor': airline_name_from_api,
                        'airline_code': airline_code,
                        'airport_code': item.get('airportcode'),
                        'scheduled_time': scheduled_time,
                        'first_date': first_date,
                        'last_date': last_date,
                        'direction': direction_kr,
                        'season': item.get('season'),
                        'monday': item.get('monday', 'N').upper() == 'Y',
                        'tuesday': item.get('tuesday', 'N').upper() == 'Y',
                        'wednesday': item.get('wednesday', 'N').upper() == 'Y',
                        'thursday': item.get('thursday', 'N').upper() == 'Y',
                        'friday': item.get('friday', 'N').upper() == 'Y',
                        'saturday': item.get('saturday', 'N').upper() == 'Y',
                        'sunday': item.get('sunday', 'N').upper() == 'Y'
                    })

                if not current_page_data:
                    print(f"⚠️ 페이지 {page_no}에 삽입할 데이터가 없습니다.")
                    break

                all_data.extend(current_page_data)

                total_count = body.get('totalCount', 0)
                print(f"현재까지 {len(all_data)}건 수집 (총 {total_count}건 예상). 다음 페이지...")

                if len(all_data) >= total_count and len(items) < num_of_rows:
                    break

                page_no += 1

            except requests.exceptions.RequestException as e:
                print(f"❌ API 요청 오류: {e}")
                break
            except Exception as e:
                print(f"❌ 데이터 처리 오류: {e}")
                if 'json_data' in locals():
                    print(f"전체 응답: {json.dumps(json_data, indent=2, ensure_ascii=False)}")
                break

    if all_data:
        df = pd.DataFrame(all_data)
        for col in df.columns:
            if df[col].dtype == 'object':
                df[col] = df[col].apply(lambda x: None if (pd.isna(x) or (isinstance(x, str) and x.strip() == '')) else x.strip() if isinstance(x, str) else x)

        data_to_insert = df.to_dict(orient="records")

        tmp_collection = db[tmp_collection_name]
        tmp_collection.delete_many({})

        if data_to_insert:
            tmp_collection.insert_many(data_to_insert)
            print(f"✅ 임시 컬렉션 '{tmp_collection_name}'에 총 {len(data_to_insert)}건 삽입 완료.")

            if collection_name in db.list_collection_names():
                db[collection_name].drop()
                print(f"⚠️ 기존 원본 컬렉션 '{collection_name}' 삭제 완료.")

            tmp_collection.rename(collection_name)
            print(f"✅ 임시 컬렉션 이름을 '{collection_name}'으로 변경 완료.")
        else:
            print("⚠️ 삽입할 데이터가 없습니다.")
    else:
        print("⚠️ 삽입할 데이터가 없습니다.")
        
    return missing_airlines_overall

    client.close()

def load_airline_mapping():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError(".env 파일에서 MONGO_URI를 찾을 수 없습니다.")

    try:
        client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        db = client["AirBot"]

        print("\n--- 항공사명-코드 매핑 정보 로드 중 ---")
        airline_collection = db["Airline"]
        airlines = airline_collection.find({}, {'_id': 0, 'airline_name_kor': 1, 'airline_code': 1})

        airline_map = {}
        for airline in airlines:
            name = airline.get('airline_name_kor')
            code = airline.get('airline_code')
            if name and code:
                airline_map[name.strip()] = code.strip()

        print(f"✅ 항공사 매핑 정보 로드 완료. 총 {len(airline_map)}건.")
        return airline_map

    finally:
        client.close()

def fetch_flight_schedule():
    
    airline_mapping_dict = load_airline_mapping()
    
    if not airline_mapping_dict:
        raise RuntimeError("항공사 매핑 정보를 로드할 수 없습니다.")

    missing_airlines_overall = fetch_and_upload_flight_schedule()
    
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    CSV_FILES_DIR = os.path.join(BASE_DIR, "output")

    if missing_airlines_overall:
        output_directory = CSV_FILES_DIR
        os.makedirs(output_directory, exist_ok=True)

        output_csv_path = os.path.join(CSV_FILES_DIR, "unmatched_airlines_for_schedules.csv")
        df_missing = pd.DataFrame(sorted(list(missing_airlines_overall)), columns=['airline_name_from_schedule_api'])
        df_missing.to_csv(output_csv_path, index=False, encoding='utf-8-sig')

        print(f"\n✅ 매핑되지 않은 항공사 목록이 '{output_csv_path}'에 저장되었습니다. 총 {len(missing_airlines_overall)}개.")
    else:
        print("\n✅ 모든 항공사가 성공적으로 매핑되었습니다.")
