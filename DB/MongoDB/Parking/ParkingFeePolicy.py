import os
import json
import requests
import re
import pandas as pd
from collections import defaultdict
from dotenv import load_dotenv
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from ..Key.key_manager import get_valid_api_key

load_dotenv()

PARKING_FEE_API_KEY = os.getenv("PARKING_FEE_API_KEY")
PARKING_FEE_API_URL = "http://apis.data.go.kr/B551177/ParkingChargeInfo/getParkingChargeInformation"


MONGO_URI = os.getenv("MONGO_URI") 
db_name = 'AirBot' 
collection_name = 'ParkingFeePolicy' 

if not MONGO_URI:
    print("경고: MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인해주세요. MongoDB 업로드를 건너뜁니다.")


params = {
    'numOfRows': '100',
    'pageNo': '1',
    'type': 'json'
}

charge_name_map = {
    "NF00000001": "단기주차", "NF00000002": "장기주차 소형차", "NF00000003": "장기주차 대형차",
    "NF00000004": "소형 화물차", "NF00000005": "대형 화물차", "NF00000007": "대형 화물차(할인)",
    "NF00000008": "대형 화물차(할인X)", "NF00000009": "주차대행(프리미엄)", "NF00000010": "예약P5(하얏트전면)",
    "NF00000011": "T2 청사", "NF00000012": "제2물류단지 소형차", "NF00000013": "제2물류단지 대형차",
    "FB00000001": "단기주차", "FB00000002": "장기주차 소형차", "FB00000003": "장기주차 대형차",
    "FB00000004": "소형 화물차", "FB00000005": "대형 화물차", "FB00000006": "무료",
    "FB00000007": "대형 화물차(할인)", "FB00000008": "주차대행(프리미엄)", "FB00000009": "예약P5(하얏트전면)",
    "FB00000010": "예약P5(하얏트전면) 무료", "FB00000011": "T2청사(무료)", "FB00000012": "T2청사(유료)",
    "FB00000013": "제2물류단지 소형차", "FB00000014": "제2물류단지 대형차"
}


def time_to_minutes(time_str):
    if not time_str or str(time_str).strip() == '-':
        return None
    parts = str(time_str).split(':')
    if len(parts) == 2:
        try:
            return int(parts[0]) * 60 + int(parts[1])
        except ValueError:
            return None
    return None

def price_to_int(price_str):
    if not price_str or str(price_str).strip() == '-':
        return None
    try:
        return int(str(price_str).replace('원', '').strip())
    except ValueError:
        return None

def parse_chardesc_for_calculation(chardesc_text):
    initial_duration_minutes = None
    initial_price_krw = None
    extra_unit_duration_minutes = None
    extra_unit_price_krw = None
    daily_max_price_krw = None
    is_free = False

    chardesc_text = str(chardesc_text) if chardesc_text is not None else ""

    match_initial = re.search(r'최초 (\d{2}:\d{2}) 에 한해 (\d+원) 적용', chardesc_text)
    if match_initial:
        initial_duration_minutes = time_to_minutes(match_initial.group(1))
        initial_price_krw = price_to_int(match_initial.group(2))
        if initial_price_krw == 0:
            is_free = True
        
    match_extra = re.search(r'(\d{2}:\d{2}) 초과 시 (\d+원) 부과', chardesc_text)
    if match_extra:
        extra_unit_duration_minutes = time_to_minutes(match_extra.group(1))
        extra_unit_price_krw = price_to_int(match_extra.group(2))
        if extra_unit_price_krw == 0:
            is_free = True

    match_daily_max = re.search(r'일일 최대 (\d+원) 적용', chardesc_text)
    if match_daily_max:
        daily_max_price_krw = price_to_int(match_daily_max.group(1))
        if daily_max_price_krw == 0:
            is_free = True

    if "무료" in chardesc_text:
        is_free = True
        initial_duration_minutes = 0
        initial_price_krw = 0
        extra_unit_duration_minutes = 0
        extra_unit_price_krw = 0
        daily_max_price_krw = 0
        
    match_after = re.search(r'이후 (\d+원)', chardesc_text)
    if match_after and extra_unit_price_krw is None:
        extra_unit_price_krw = price_to_int(match_after.group(1))
        extra_unit_duration_minutes = 1
        if extra_unit_price_krw == 0:
            is_free = True

    if is_free:
        initial_duration_minutes = initial_duration_minutes if initial_duration_minutes is not None else 0
        initial_price_krw = 0
        extra_unit_duration_minutes = extra_unit_duration_minutes if extra_unit_duration_minutes is not None else 0
        extra_unit_price_krw = 0
        daily_max_price_krw = 0

    return initial_duration_minutes, initial_price_krw, \
           extra_unit_duration_minutes, extra_unit_price_krw, \
           daily_max_price_krw, is_free

# policy_title에서 parking_type과 car_type을 추출하는 함수
def categorize_policy_title(policy_title):
    parking_type = None
    car_type = None

    # Determine parking_type
    if "단기" in policy_title:
        parking_type = "단기"
    elif "장기" in policy_title:
        parking_type = "장기"
    elif "화물" in policy_title:
        parking_type = "화물"
    elif "주차대행" in policy_title:
        parking_type = "주차대행"
    elif "예약" in policy_title:
        parking_type = "예약"
    elif policy_title == "무료": 
        parking_type = "무료"
    
    # Determine car_type
    if "소형" in policy_title:
        car_type = "소형"
    elif "대형" in policy_title:
        car_type = "대형"
    
    return parking_type, car_type


try:
    print("API 데이터 요청 시도.")
    
    # type='public' 키 요청
    service_key = get_valid_api_key(PARKING_FEE_API_URL, params, key_type="public", auth_param_name="serviceKey")

    if not service_key:
        print("유효한 API 키를 찾지 못해 작업을 종료합니다.")
        exit

    params = params.copy()
    params['serviceKey'] = service_key    
    
    response = requests.get(PARKING_FEE_API_URL, params=params)
    response.raise_for_status()
    data = response.json()

    items = data.get('response', {}).get('body', {}).get('items', [])
    print(f"API에서 {len(items)}개의 아이템을 성공적으로 가져왔습니다.")

    merged_data = defaultdict(lambda: {
        'initial_duration_minutes': None,
        'initial_price_krw': None,
        'extra_unit_duration_minutes': None,
        'extra_unit_price_krw': None,
        'daily_max_price_krw': None,
        'is_free': False,
        'datetime': None 
    })

    for item in items:
        charid = item.get('charid')
        chardesc = item.get('chardesc')
        datetime_val = item.get('datetime')

        chargename = charge_name_map.get(charid, charid)

        initial_dur, initial_price, \
        extra_dur, extra_price, \
        daily_max_price, is_free = parse_chardesc_for_calculation(chardesc)

        current_charge = merged_data[chargename]

        current_charge['is_free'] = current_charge['is_free'] or is_free
        
        if initial_dur is not None and current_charge['initial_duration_minutes'] is None:
            current_charge['initial_duration_minutes'] = initial_dur
        if initial_price is not None and current_charge['initial_price_krw'] is None:
            current_charge['initial_price_krw'] = initial_price
        
        if extra_dur is not None and current_charge['extra_unit_duration_minutes'] is None:
            current_charge['extra_unit_duration_minutes'] = extra_dur
        if extra_price is not None and current_charge['extra_unit_price_krw'] is None:
            current_charge['extra_unit_price_krw'] = extra_price
            
        if daily_max_price is not None and current_charge['daily_max_price_krw'] is None:
            current_charge['daily_max_price_krw'] = daily_max_price
            
        if datetime_val is not None:
            current_charge['datetime'] = datetime_val


    processed_data_for_mongo = []

    output_column_mapping = {
        "policy_title_source": "policy_title", 
        "initial_duration_minutes": "inital_dueation_minutes",
        "initial_price_krw": "initial_price_krw",
        "extra_unit_duration_minutes": "extra_unit_duration_minutes",
        "extra_unit_price_krw": "extra_unit_price_krw",
        "daily_max_price_krw": "daily_max_price_krw",
        "is_free": "is_free",
        "parking_type": "parking_type",
        "car_type": "car_type"
    }
    
    output_columns_ordered = [
        "policy_title", 
        "inital_dueation_minutes",
        "initial_price_krw",
        "extra_unit_duration_minutes",
        "extra_unit_price_krw",
        "daily_max_price_krw",
        "is_free",
        "parking_type",
        "car_type"
    ]


    for chargename, values in merged_data.items():
        policy_title_val = chargename 
        parking_type_val, car_type_val = categorize_policy_title(policy_title_val)

        intermediate_row_data = {
            "policy_title_source": policy_title_val, 
            "initial_duration_minutes": values['initial_duration_minutes'],
            "initial_price_krw": values['initial_price_krw'],
            "extra_unit_duration_minutes": values['extra_unit_duration_minutes'],
            "extra_unit_price_krw": values['extra_unit_price_krw'],
            "daily_max_price_krw": values['daily_max_price_krw'],
            "is_free": values['is_free'],
            "parking_type": parking_type_val,
            "car_type": car_type_val
        }
        
        final_row = {output_column_mapping[key]: intermediate_row_data[key] for key in output_column_mapping}
        
        processed_data_for_mongo.append(final_row)

    print(f"\n전처리된 데이터 {len(processed_data_for_mongo)}개 준비 완료.")
    
    print("\n--- 전처리된 첫 번째 행 확인 (업로드 전) ---")
    if processed_data_for_mongo:
        print(processed_data_for_mongo[0])
    else:
        print("전처리된 데이터가 없습니다.")
    print("-------------------------------------------\n")

    if processed_data_for_mongo:
        df = pd.DataFrame(processed_data_for_mongo, columns=output_columns_ordered)
        
        numeric_columns_to_fill_with_0 = [
            "Inital_dueation_minutes",
            "Initial_price_krw",
            "extra_unit_duration_minutes",
            "extra_unit_price_krw",
            "daily_max_price_krw"
        ]
        for col in numeric_columns_to_fill_with_0:
            if col in df.columns:
                df[col] = df[col].fillna(0) 
        

        if MONGO_URI: 
            client = None
            try:
                client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
                client.admin.command('ping')
                print("MongoDB Atlas에 성공적으로 연결되었습니다.")

                db = client[db_name] 
                collection = db[collection_name] 

                records_to_upload = df.to_dict('records')
                
                inserted_count = 0
                updated_count = 0

                for doc in records_to_upload:
                    if not doc.get("policy_title"): 
                        print(f"경고: 'policy_title'이 없는 문서는 MongoDB에 업로드되지 않습니다. {doc}")
                        continue
                    
                    result = collection.update_one(
                        {"policy_title": doc["policy_title"]}, 
                        {"$set": doc},
                        upsert=True
                    )
                    if result.upserted_id:
                        inserted_count += 1
                    elif result.modified_count:
                        updated_count += 1

                print(f"\nMongoDB '{db_name}.{collection_name}' 컬렉션에 {inserted_count}개의 문서 삽입, {updated_count}개의 문서 업데이트 완료.")
            except Exception as e:
                print(f"\nMongoDB 업로드 중 오류 발생: {e}")
            finally:
                if client:
                    client.close()
                    print("MongoDB 연결이 닫혔습니다.")
        else:
            print("\nMONGO_URI 환경 변수가 설정되지 않아, MongoDB 업로드를 건너뛰었습니다. .env 파일을 확인해주세요.")
            
        print(f"총 {len(processed_data_for_mongo)}개의 고유한 주차 요금 정책이 처리되었습니다.")
    else:
        print("MongoDB에 업로드할 데이터가 없습니다.")

except requests.exceptions.RequestException as e:
    print(f"API 요청 중 오류가 발생했습니다: {e}")
except json.JSONDecodeError:
    print("API 응답이 유효한 JSON 형식이 아닙니다.")
except Exception as e:
    print(f"데이터 처리 중 알 수 없는 오류가 발생했습니다: {e}")