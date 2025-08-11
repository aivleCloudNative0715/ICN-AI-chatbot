import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

def upload_airport_data_atomic(file_path):
    """
    CSV 파일을 읽어 임시 컬렉션에 저장 후 기존 'Airport' 컬렉션과 교체하는 방식으로 안전하게 업로드합니다.
    """
    load_dotenv()
    mongo_uri = os.getenv("MONGO_URI")

    if not mongo_uri:
        raise ValueError(".env 파일에서 MONGO_URI를 찾을 수 없습니다.")

    final_collection_name = "Airport"
    temp_collection_name = "Airport_tmp"

    client = MongoClient(mongo_uri, server_api=ServerApi('1'))

    try:
        db = client["AirBot"]
        client.admin.command('ping')
        print("MongoDB에 성공적으로 연결되었습니다!")

        df = pd.read_csv(file_path, encoding='cp949')
        df_selected = df[['공항코드1(IATA)', '한글공항', '한글국가명']].copy()
        df_selected.rename(columns={
            '공항코드1(IATA)': 'airport_code',
            '한글공항': 'airport_name_kor',
            '한글국가명': 'country_name_kor'
        }, inplace=True)

        for col in df_selected.columns:
            df_selected[col] = df_selected[col].apply(lambda x: None if pd.isna(x) else str(x))

        data = df_selected.to_dict(orient="records")

        temp_collection = db[temp_collection_name]
        # 임시 컬렉션 초기화 및 데이터 삽입
        temp_collection.delete_many({})
        temp_collection.insert_many(data)
        print(f"임시 컬렉션 '{temp_collection_name}'에 {len(data)}건 삽입 완료")

        # 기존 컬렉션 삭제 후 임시 컬렉션 이름 변경
        if final_collection_name in db.list_collection_names():
            db[final_collection_name].drop()
            print(f"기존 컬렉션 '{final_collection_name}' 삭제 완료")

        temp_collection.rename(final_collection_name)
        print(f"임시 컬렉션을 '{final_collection_name}'로 이름 변경 완료")
        print("데이터 갱신이 안전하게 완료되었습니다.")

    except FileNotFoundError:
        print(f"파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        client.close()
        print("MongoDB 연결이 닫혔습니다.")

if __name__ == "__main__":
    excel_files_dir = "./db"
    airport_list_file = os.path.join(excel_files_dir, "국토교통부_세계공항_정보_20241231.csv")

    print("\n--- '국토교통부_세계공항_정보_20241231.csv' 데이터 업로드 시작 ---")
    upload_airport_data_atomic(airport_list_file)
    print("\n--- 데이터 업로드 완료 및 MongoDB 연결 종료 ---")
