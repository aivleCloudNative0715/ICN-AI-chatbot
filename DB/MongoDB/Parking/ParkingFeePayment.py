import os
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

def upload_payment_methods_from_csv(csv_file_path="ParkingFeePayment.csv"):
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

    client = None
    try:
        client = MongoClient(MONGO_URI, server_api=ServerApi('1'))
        client.admin.command('ping')
        print("MongoDB Atlas에 성공적으로 연결되었습니다.")

        db = client['AirBot']
        collection_name = 'ParkingFeePayment'
        temp_collection_name = collection_name + "_temp"
        collection_temp = db[temp_collection_name]

        print(f"CSV 파일 '{csv_file_path}' 읽기 시도.")
        df_payment = pd.read_csv(csv_file_path)
        print(f"CSV 파일에서 {len(df_payment)}개의 행을 성공적으로 읽었습니다.")

        processed_payment_documents = []

        boolean_cols = ['available_cash', 'available_prepaid', 'available_postpaid',
                        'available_credit', 'available_transit', 'available_hipass']

        for index, row in df_payment.iterrows():
            doc = {
                "payment_title": row.get('payment_title'),
                "payment_step_description": row.get('payment_step_description'),
            }

            for col in boolean_cols:
                csv_value = str(row.get(col, '')).lower().strip()
                doc[col] = (csv_value == 'true')

            processed_payment_documents.append(doc)

        print(f"\n전처리된 데이터 {len(processed_payment_documents)}개 준비 완료.")

        if processed_payment_documents:
            print("🔎 전처리된 샘플 문서:")
            print(processed_payment_documents[0])
        else:
            print("전처리된 데이터가 없습니다.")

        # 임시 컬렉션 초기화
        collection_temp.delete_many({})

        if processed_payment_documents:
            collection_temp.insert_many(processed_payment_documents)

            # 기존 컬렉션 삭제 후 임시 컬렉션 이름 변경
            if collection_name in db.list_collection_names():
                db.drop_collection(collection_name)
            db[temp_collection_name].rename(collection_name)

            print(f"\nMongoDB에 {len(processed_payment_documents)}개 문서가 저장되었습니다.")
        else:
            print("저장할 PaymentMethod 데이터가 없습니다.")

    except FileNotFoundError:
        print(f"오류: CSV 파일 '{csv_file_path}'를 찾을 수 없습니다.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        if client:
            client.close()
            print("MongoDB 연결이 닫혔습니다.")

if __name__ == "__main__":
    upload_payment_methods_from_csv()
