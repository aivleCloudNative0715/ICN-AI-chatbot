import os
import pandas as pd
from pymongo.mongo_client import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv

load_dotenv()

def upload_parking_lot_from_csv(csv_file_path):
    MONGO_URI = os.getenv("MONGO_URI")
    if not MONGO_URI:
        raise ValueError("MONGO_URI 환경 변수가 설정되지 않았습니다. .env 파일을 확인하세요.")

    client = MongoClient(MONGO_URI, server_api=ServerApi('1'))

    try:
        client.admin.command('ping')
        print("MongoDB Atlas에 성공적으로 연결되었습니다.")

        db = client['AirBot']
        collection = db['ParkingLot']

        print(f"CSV 파일 '{csv_file_path}' 읽기 시도.")
        df = pd.read_csv(csv_file_path)

        print(f"{len(df)}개의 행을 성공적으로 읽었습니다.")

        df = df.where(pd.notnull(df), None)

        processed_documents = []
        for _, row in df.iterrows():
            doc = {
                "parking_type": row.get("parking_type"),
                "floor": row.get("floor"),
                "zone": row.get("zone"),
                "terminal": row.get("terminal")
            }
            processed_documents.append(doc)

        print(f"\n총 {len(processed_documents)}개의 문서가 준비되었습니다.")
        print("\n--- 첫 번째 문서 확인 ---")
        if processed_documents:
            print(processed_documents[0])
        else:
            print("전처리된 문서가 없습니다.")
        print("--------------------------\n")

        if processed_documents:
            result = collection.insert_many(processed_documents)
            print(f"{len(result.inserted_ids)}개의 문서가 ParkingLot 컬렉션에 성공적으로 삽입되었습니다.")
        else:
            print("삽입할 문서가 없습니다.")

    except FileNotFoundError:
        print(f"오류: CSV 파일 '{csv_file_path}'를 찾을 수 없습니다. 파일 경로를 확인하세요.")
    except Exception as e:
        print(f"오류 발생: {e}")
    finally:
        client.close()
        print("MongoDB 연결이 닫혔습니다.")

if __name__ == "__main__":
    # 예시: 절대경로 혹은 프로젝트 루트 기준 상대경로 전달
    example_path = os.path.join("C:", "Users", "User", "Desktop", "ICN-AI-chatbot", "DB", "MongoDB", "Parking", "ParkingLot.csv")
    upload_parking_lot_from_csv(example_path)
