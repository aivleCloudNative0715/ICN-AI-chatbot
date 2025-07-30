import requests
import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from ..Key.key_manager import get_valid_api_key

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수에서 MongoDB URI와 API 서비스 키를 가져옵니다.
mongo_uri = os.getenv("MONGO_URI")

if not mongo_uri:
    print("❌ 오류: .env 파일에서 MONGO_URI를 찾을 수 없습니다. 파일을 확인해주세요.")
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
AIRLINE_API_URL = "http://apis.data.go.kr/B551177/StatusOfSrvAirlines/getServiceAirlineInfo"

# --- '취항 항공사 현황 정보' API에서 데이터를 가져와 'Airline' 컬렉션에 업로드하는 함수 ---
def upload_airline_data_from_api(db):
    
  
    """
    취항 항공사 현황 정보 API를 호출하여 데이터를 가져와 'Airline' 컬렉션에 삽입합니다.
    airline_code (IATA), airline_name_kor, airline_contact (공항연락처)를 추출합니다.
    """
    collection_name = "Airline"
    try:
        params = {
            'type': 'json' # 응답 타입을 JSON으로 명시
            # 'airline_iata': '', # 모든 항공사를 가져오려면 이 파라미터는 제거하거나 비워둡니다.
            # 'airline_icao': ''
        }
        
        
        # type='public' 키 요청
        service_key = get_valid_api_key(AIRLINE_API_URL, params, key_type="public", auth_param_name="serviceKey")

        if not service_key:
            print("유효한 API 키를 찾지 못해 작업을 종료합니다.")
            return

        params = params.copy()
        params['serviceKey'] = service_key
            
        
        print(f"API 요청 중: {AIRLINE_API_URL} (params: {params})")
        response = requests.get(AIRLINE_API_URL, params=params)
        response.raise_for_status() # HTTP 오류 (4xx, 5xx)가 발생하면 예외 발생
        
        json_data = response.json()

        items = json_data.get('response', {}).get('body', {}).get('items', [])
        
        if not items:
            print("⚠️ API 응답에 'items' 데이터가 없거나 비어 있습니다.")
            # 성공했지만 데이터가 없는 경우를 고려하여 return
            # resultCode/resultMsg 확인 로직을 추가하여 '정상 서비스' 여부를 판단할 수도 있습니다.
            result_code = json_data.get('response', {}).get('header', {}).get('resultCode')
            result_msg = json_data.get('response', {}).get('header', {}).get('resultMsg')
            print(f"API 응답 결과: [Result Code: {result_code}, Result Message: {result_msg}]")
            return

        extracted_data = []
        for item in items:
            extracted_data.append({
                'airline_code': item.get('airlineIata'),    # 항공사 IATA 코드
                'airline_name_kor': item.get('airlineName'), # 항공사 한글명
                'airline_contact': item.get('airlineIcTel') # 공항연락처
            })
        
        if not extracted_data:
            print("⚠️ 추출된 데이터가 없습니다. 필드명이 정확한지 확인해주세요.")
            return

        df = pd.DataFrame(extracted_data)

        # 추출된 컬럼들의 NaN 값을 None으로 변환 (API 응답에서 null/빈 문자열이 올 수 있으므로)
        for col in df.columns:
            df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x).strip() if isinstance(x, str) else x)

        data = df.to_dict(orient="records")

        collection = db[collection_name]
        # collection.delete_many({}) # 기존 데이터를 삭제하고 새로 넣고 싶을 때만 주석 해제

        collection.insert_many(data)
        print(f"✅ 성공적으로 API 데이터를 '{collection_name}' 컬렉션에 삽입했습니다. 총 {len(data)}건.")
    
    except requests.exceptions.RequestException as e:
        print(f"❌ API 요청 중 오류가 발생했습니다: {e}")
    except Exception as e:
        print(f"❌ 데이터 처리 중 오류가 발생했습니다: {e}")
        # 디버깅을 위해 전체 응답 출력 (오류 발생 시에만)
        if 'json_data' in locals(): # json_data 변수가 정의되어 있을 경우에만 출력
            print(f"디버깅용 전체 응답: {json_data}")


# --- 메인 실행 부분: Airline 데이터만 업로드합니다. ---

print("\n--- '취항 항공사 현황 정보' API 데이터 업로드 시작 ---")

upload_airline_data_from_api(db)

# --- MongoDB 연결 종료 ---
client.close()
print("\n--- 데이터 업로드 완료 및 MongoDB 연결 종료 ---")