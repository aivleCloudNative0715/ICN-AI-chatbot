import pandas as pd
import requests
from datetime import datetime
from pymongo import MongoClient
from dotenv import load_dotenv
import os
from ..Key.key_manager import get_valid_api_key

load_dotenv()

def fetch_and_save_atmos_data():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        print("오류: .env 파일에서 MONGO_URI를 찾을 수 없습니다. 파일을 확인해주세요.")
        return

    client = None
    try:
        client = MongoClient(mongo_uri)
        db = client["AirBot"]
        collection_name = "ATMOS"
        temp_collection_name = collection_name + "_temp"
        collection_temp = db[temp_collection_name]

        url = 'https://apihub.kma.go.kr/api/typ01/url/amos.php'
        params_base = {
            'dtm': '10',
            'stn': '113',
            'help': '0'
        }

        authKey = get_valid_api_key(url, params_base, key_type="weather", auth_param_name="authKey")
        if not authKey:
            print("유효한 API 키를 찾지 못해 작업을 종료합니다.")
            return

        params = params_base.copy()
        params['authKey'] = authKey

        current_time_str = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        print(f"[{current_time_str}] ATMOS API 요청 시작...")

        response = requests.get(url, params=params)
        response.raise_for_status()

        try:
            text = response.content.decode('euc-kr')
        except UnicodeDecodeError:
            print(f"[{current_time_str}] 'euc-kr' 디코딩 실패. 'cp949'로 재시도합니다.")
            try:
                text = response.content.decode('cp949')
            except UnicodeDecodeError as e:
                print(f"[{current_time_str}] 디코딩 실패: {e}")
                return

        split_marker = "#START7777"
        if split_marker not in text:
            print(f"[{current_time_str}] split_marker not found.")
            return

        after_marker = text.split(split_marker, 1)[1]
        lines = after_marker.strip().splitlines()
        data_lines = [line for line in lines if line.strip() and not line.strip().startswith("#")]
        if not data_lines:
            print(f"[{current_time_str}] 파싱할 데이터 라인이 없습니다.")
            return

        data = [line.strip().split() for line in data_lines]

        columns = [
            "Region", "Datetime", "L_VIS", "R_VIS", "L_RVR", "R_RVR", "CH_MIN",
            "TA", "TD", "HM", "PS", "PA", "RN", "Reserve1", "Reserve2",
            "WD02", "WD02_MAX", "WD02_MIN", "WS02", "WS02_MAX", "WS02_MIN",
            "WD10", "WD10_MAX", "WD10_MIN", "WS10", "WS10_MAX", "WS10_MIN"
        ]
        df = pd.DataFrame(data, columns=columns)
        df['Datetime'] = pd.to_datetime(df['Datetime'], format='%Y%m%d%H%M', errors='coerce')

        df_to_save = df[['Datetime', 'L_VIS', 'TA', 'HM', 'RN', 'WS10']].copy()
        df_to_save.columns = ['tm', 'l_vis', 'ta', 'hm', 'rn', 'ws_10']

        # 임시 컬렉션 초기화
        collection_temp.delete_many({})

        documents = df_to_save.to_dict(orient='records')

        if documents:
            collection_temp.insert_many(documents)
            # 기존 컬렉션 삭제 후 임시 컬렉션 rename
            if collection_name in db.list_collection_names():
                db.drop_collection(collection_name)
            collection_temp.rename(collection_name)

            print(f"[{current_time_str}] MongoDB 저장 완료. 총 {len(documents)}개 문서.")
        else:
            print(f"[{current_time_str}] 저장할 문서가 없습니다.")

    except Exception as e:
        print(f"[{current_time_str}] 오류 발생: {e}")

    finally:
        if client:
            client.close()


if __name__ == "__main__":
    fetch_and_save_atmos_data()
