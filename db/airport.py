import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

# .env 파일에서 환경 변수를 로드합니다.
load_dotenv()

# 환경 변수에서 MongoDB URI를 가져옵니다.
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

# --- 엑셀 파일 경로 설정 ---
EXCEL_FILES_DIR = "./db"
airport_list_file = os.path.join(EXCEL_FILES_DIR, "국토교통부_세계공항_정보_20241231.csv")


# --- '국토교통부_세계공항_정보_20241231.csv' 파일을 'Airport' 컬렉션으로 업로드하는 함수 ---
def upload_airport_data(file_path, db):
    """
    '국토교통부_세계공항_정보_20241231.csv' 파일을 읽어와 'Airport' 컬렉션에 데이터를 삽입합니다.
    airport_code (IATA), airport_name_kor (한글공항명), 한글국가명만 추출합니다.
    """
    collection_name = "Airport"
    try:
        df = pd.read_csv(file_path, encoding='cp949')

        # 필요한 컬럼만 선택하고 이름을 변경합니다.
        # '공항코드1(IATA)', '한글공항', '한글국가명' 만 선택
        df_selected = df[['공항코드1(IATA)', '한글공항', '한글국가명']].copy()
        df_selected.rename(columns={
            '공항코드1(IATA)': 'airport_code',
            '한글공항': 'airport_name_kor',
            '한글국가명': 'country_name_kor' # 한글국가명 필드 추가
        }, inplace=True)

        # 추출된 모든 컬럼들의 NaN 값을 None으로 변환하여 MongoDB의 null로 저장
        for col in df_selected.columns:
            df_selected[col] = df_selected[col].apply(lambda x: None if pd.isna(x) else str(x))

        data = df_selected.to_dict(orient="records")

        collection = db[collection_name]
        # collection.delete_many({}) # 기존 데이터를 삭제하고 새로 넣고 싶을 때만 주석 해제

        collection.insert_many(data)
        print(f"✅ 성공적으로 '{file_path}' 데이터를 '{collection_name}' 컬렉션에 삽입했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
    except Exception as e:
        print(f"❌ '{file_path}' 처리 중 오류가 발생했습니다: {e}")


# --- 메인 실행 부분: Airport 데이터만 업로드합니다. ---

print("\n--- '국토교통부_세계공항_정보_20241231.csv' 데이터 업로드 시작 ---")

upload_airport_data(airport_list_file, db)

# --- MongoDB 연결 종료 ---
client.close()
print("\n--- 데이터 업로드 완료 및 MongoDB 연결 종료 ---")