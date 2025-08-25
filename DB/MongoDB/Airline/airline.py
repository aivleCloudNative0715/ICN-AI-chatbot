import requests
import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os
from ..Key.key_manager import get_valid_api_key

def upload_airline_data_atomic():
    """
    '취항 항공사 현황 정보' API에서 데이터를 받아 임시 컬렉션에 저장 후,
    기존 'Airline' 컬렉션을 삭제하고 임시 컬렉션을 'Airline'으로 이름 변경하는 방식으로 안전하게 갱신합니다.
    """
    load_dotenv()
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError(".env 파일에서 MONGO_URI를 찾을 수 없습니다.")

    AIRLINE_API_URL = "http://apis.data.go.kr/B551177/StatusOfSrvAirlines/getServiceAirlineInfo"
    final_collection_name = "Airline"
    temp_collection_name = "Airline_tmp"

    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

    try:
        db = client["AirBot"]
        client.admin.command('ping')
        print("MongoDB에 성공적으로 연결되었습니다!")

        params = {'type': 'json'}
        service_key = get_valid_api_key(
            AIRLINE_API_URL, params,
            key_type="public",
            auth_param_name="serviceKey"
        )
        if not service_key:
            print("유효한 API 키를 찾지 못해 작업을 종료합니다.")
            return

        params['serviceKey'] = service_key
        print(f"API 요청 중: {AIRLINE_API_URL} (params: {params})")
        response = requests.get(AIRLINE_API_URL, params=params)
        response.raise_for_status()
        json_data = response.json()

        items = json_data.get('response', {}).get('body', {}).get('items', [])
        if not items:
            print("API 응답에 'items' 데이터가 없거나 비어 있습니다.")
            result_code = json_data.get('response', {}).get('header', {}).get('resultCode')
            result_msg = json_data.get('response', {}).get('header', {}).get('resultMsg')
            print(f"API 응답 결과: [Result Code: {result_code}, Result Message: {result_msg}]")
            return

        extracted_data = [
            {
                'airline_code': item.get('airlineIata'),
                'airline_name_kor': item.get('airlineName'),
                'airline_contact': item.get('airlineIcTel')
            }
            for item in items
        ]

        if not extracted_data:
            print("추출된 데이터가 없습니다. 필드명을 확인해주세요.")
            return

        df = pd.DataFrame(extracted_data)

        # NaN → None, 문자열은 strip()
        for col in df.columns:
            df[col] = df[col].apply(
                lambda x: None if pd.isna(x) else str(x).strip() if isinstance(x, str) else x
            )

        data = df.to_dict(orient="records")

        temp_collection = db[temp_collection_name]

        # 임시 컬렉션 초기화 후 삽입
        temp_collection.delete_many({})
        temp_collection.insert_many(data)
        print(f"임시 컬렉션 '{temp_collection_name}'에 {len(data)}건 삽입 완료")

        # 기존 컬렉션 삭제 및 임시 컬렉션 이름 변경
        if final_collection_name in db.list_collection_names():
            db[final_collection_name].drop()
            print(f"기존 컬렉션 '{final_collection_name}' 삭제 완료")

        temp_collection.rename(final_collection_name)
        print(f"임시 컬렉션을 '{final_collection_name}'로 이름 변경 완료")
        print("데이터 갱신이 안전하게 완료되었습니다.")

    except requests.exceptions.RequestException as e:
        print(f"API 요청 오류: {e}")
    except Exception as e:
        print(f"데이터 처리 오류: {e}")
        if 'json_data' in locals():
            print(f"디버깅용 전체 응답: {json_data}")
    finally:
        client.close()
        print("MongoDB 연결이 닫혔습니다.")

if __name__ == "__main__":
    upload_airline_data_atomic()
