import os
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

def upload_discount_policy_from_csv(csv_file_path="ParkingFeeDiscountPolicy.csv"):
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

    client = None
    try:
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        client.admin.command('ping')
        print("MongoDB Atlas에 성공적으로 연결되었습니다.")

        db = client['AirBot']
        collection_name = 'ParkingFeeDiscountPolicy'
        temp_collection_name = collection_name + "_temp"
        collection_temp = db[temp_collection_name]

        print(f"CSV 파일 '{csv_file_path}' 읽기 시도.")
        df_discount = pd.read_csv(csv_file_path)
        print(f"CSV 파일에서 {len(df_discount)}개의 행을 성공적으로 읽었습니다.")

        # 컬럼명 앞뒤 공백 및 보이지 않는 문자 제거
        df_discount.columns = df_discount.columns.str.strip().str.replace('\u200b', '', regex=False)

        processed_discount_documents = []

        for index, row in df_discount.iterrows():
            discount_rate = pd.to_numeric(row.get('discount_rate', None), errors='coerce')
            if pd.isna(discount_rate):
                discount_rate = None

            doc = {
                "discount_policy_title": row.get('discount_policy_title'),
                "discount_condition": row.get('discount_condition'),
                "realtime_discount_document": row.get('realtime_discount_document', None),
                "post_submission_discount_document": row.get('post_submission_discount_document', None),
                "discount_rate": discount_rate,
                "notice": row.get('notice', None),
            }
            processed_discount_documents.append(doc)

        print(f"\n전처리된 데이터 {len(processed_discount_documents)}개 준비 완료.")
        if processed_discount_documents:
            print("샘플 문서:", processed_discount_documents[0])
        else:
            print("전처리된 데이터가 없습니다.")

        # 임시 컬렉션 초기화
        collection_temp.delete_many({})

        if processed_discount_documents:
            collection_temp.insert_many(processed_discount_documents)

            # 기존 컬렉션 삭제 후 임시 컬렉션 이름 변경
            if collection_name in db.list_collection_names():
                db.drop_collection(collection_name)
            db[temp_collection_name].rename(collection_name)

            print(f"\nMongoDB에 {len(processed_discount_documents)}개 문서가 저장되었습니다.")
        else:
            print("저장할 데이터가 없습니다.")

    except FileNotFoundError:
        print(f"오류: CSV 파일 '{csv_file_path}'를 찾을 수 없습니다. 파일 경로를 확인하세요.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if client:
            client.close()
            print("MongoDB 연결이 닫혔습니다.")

if __name__ == "__main__":
    upload_discount_policy_from_csv()
