import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

def update_airline_info_atomic(csv_file_path):
    """
    CSV 파일로부터 항공사 정보를 읽어 'Airline' 컬렉션에 업데이트 또는 삽입합니다.
    MongoDB 연결 및 종료도 이 함수가 처리합니다.
    """
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError(".env 파일에서 MONGO_URI를 찾을 수 없습니다.")

    if not os.path.exists(csv_file_path):
        raise FileNotFoundError(f"지정된 CSV 파일을 찾을 수 없습니다: {csv_file_path}")

    client = MongoClient(mongo_uri, server_api=ServerApi('1'))
    try:
        db = client["AirBot"]
        client.admin.command('ping')
        print("MongoDB에 성공적으로 연결되었습니다!")

        airline_collection = db["Airline"]

        try:
            df_updates = pd.read_csv(csv_file_path)

            required_columns = ['airline_code', 'airline_name_kor', 'airline_contact']
            if not all(col in df_updates.columns for col in required_columns):
                raise ValueError(f"CSV 파일에 필요한 모든 컬럼({', '.join(required_columns)})이 없습니다. 현재 컬럼: {df_updates.columns.tolist()}")

            df_updates['airline_contact'] = df_updates['airline_contact'].replace({float('nan'): None}).mask(df_updates['airline_contact'] == '', None)

            records_to_upsert = df_updates.to_dict(orient='records')

        except pd.errors.EmptyDataError:
            print(f"CSV 파일 '{csv_file_path}'이 비어 있습니다. 업데이트할 내용이 없습니다.")
            return 0, 0, 0
        except Exception as e:
            raise RuntimeError(f"CSV 파일 읽기 오류: {e}")

        print(f"\n--- 총 {len(records_to_upsert)}건의 항공사 정보 업데이트/추가 시작 ---")

        updated_count = 0
        inserted_count = 0

        for record in records_to_upsert:
            airline_code = record.get('airline_code')
            airline_name_kor = record.get('airline_name_kor')
            airline_contact = record.get('airline_contact')

            if not airline_name_kor:
                print(f"경고: 항공사 이름이 없어 건너뜁니다: {record}")
                continue

            update_document = {
                'airline_name_kor': airline_name_kor,
                'airline_code': airline_code,
                'airline_contact': airline_contact
            }

            filter_query = {'airline_name_kor': airline_name_kor.strip()}

            result = airline_collection.update_one(
                filter_query,
                {'$set': update_document},
                upsert=True
            )

            if result.upserted_id:
                inserted_count += 1
                print(f"새 항공사 추가: '{airline_name_kor}' (IATA: {airline_code or '없음'})")
            elif result.modified_count > 0:
                updated_count += 1
                print(f"항공사 정보 업데이트: '{airline_name_kor}' (IATA: {airline_code or '없음'})")
            else:
                print(f"항공사 정보 변화 없음: '{airline_name_kor}'")

        no_change_count = len(records_to_upsert) - inserted_count - updated_count

        print(f"\n--- 항공사 정보 업데이트/추가 완료 ---")
        print(f"총 {inserted_count}건의 새 항공사 정보가 추가되었습니다.")
        print(f"총 {updated_count}건의 기존 항공사 정보가 업데이트되었습니다.")
        print(f"총 {no_change_count}건의 항공사 정보는 변경사항이 없었습니다.")

        return inserted_count, updated_count, no_change_count

    except Exception as e:
        print(f"오류 발생: {e}")
        return 0, 0, 0
    finally:
        client.close()
        print("MongoDB 연결이 닫혔습니다.")


if __name__ == "__main__":
    csv_path = "./db/output/unmatched_airlines_for_schedules.csv"
    update_airline_info_atomic(csv_path)
