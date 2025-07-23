import os
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

MONGO_URI = os.getenv("MONGO_URI")
if not MONGO_URI:
    raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

try:
    client.admin.command('ping')
    print("MongoDB Atlas에 성공적으로 연결되었습니다.")

    db = client['AirBot']
    collection_payment_method = db['ParkingFeePayment']

    csv_file_path = "ParkingFeePayment.csv" 

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
        print("⚠️ 전처리된 데이터가 없습니다.")

    if processed_payment_documents:
        inserted_count = 0
        updated_count = 0
        
        for doc in processed_payment_documents:
            if not doc.get("payment_title"):
                print(f"⚠️ 'payment_title'이 없는 문서는 건너뜁니다: {doc}")
                continue

            result = collection_payment_method.update_one(
                {"payment_title": doc["payment_title"]},
                {"$set": doc},
                upsert=True
            )
            if result.upserted_id:
                inserted_count += 1
            elif result.modified_count:
                updated_count += 1

        print(f"\n✅ MongoDB에 {inserted_count}개 문서 삽입, {updated_count}개 문서 업데이트 완료.")
    else:
        print("⚠️ 삽입할 PaymentMethod 데이터가 없습니다.")

except FileNotFoundError:
    print(f"❌ 오류: CSV 파일 '{csv_file_path}'를 찾을 수 없습니다.")
except Exception as e:
    print(f"❌ 오류 발생: {e}")
finally:
    client.close()
    print("📦 MongoDB 연결이 닫혔습니다.")
