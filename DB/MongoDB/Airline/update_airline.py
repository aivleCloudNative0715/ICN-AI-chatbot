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
    airline_collection = db["Airline"]

    client.admin.command('ping')
    print("✅ MongoDB에 성공적으로 연결되었습니다!")

except Exception as e:
    print(f"❌ MongoDB 연결 오류: {e}")
    exit(1)

# --- 업데이트할 CSV 파일 경로 ---
csv_file_path = "./db/output/unmatched_airlines_for_schedules.csv"

# CSV 파일이 존재하는지 확인합니다.
if not os.path.exists(csv_file_path):
    print(f"❌ 오류: 지정된 CSV 파일을 찾을 수 없습니다. 경로를 확인해주세요: {csv_file_path}")
    exit(1)

# --- CSV 파일 읽기 ---
try:
    # CSV 파일을 Pandas DataFrame으로 읽어옵니다.
    # 헤더가 있다고 가정하고 첫 줄을 컬럼 이름으로 사용합니다.
    df_updates = pd.read_csv(csv_file_path)

    # 필요한 컬럼이 모두 있는지 확인합니다.
    required_columns = ['airline_code', 'airline_name_kor', 'airline_contact']
    if not all(col in df_updates.columns for col in required_columns):
        print(f"❌ 오류: CSV 파일에 필요한 모든 컬럼({', '.join(required_columns)})이 없습니다.")
        print(f"현재 파일 컬럼: {df_updates.columns.tolist()}")
        exit(1)

    # 'airline_contact'가 비어있는 경우(NaN 또는 빈 문자열) None으로 처리합니다.
    df_updates['airline_contact'] = df_updates['airline_contact'].replace({float('nan'): None}).mask(df_updates['airline_contact'] == '', None)

    # 업데이트할 데이터프레임을 딕셔너리 리스트로 변환합니다.
    records_to_upsert = df_updates.to_dict(orient='records')

except pd.errors.EmptyDataError:
    print(f"⚠️ 경고: CSV 파일 '{csv_file_path}'이 비어 있습니다. 업데이트할 내용이 없습니다.")
    exit(0)
except Exception as e:
    print(f"❌ CSV 파일 읽기 오류: {e}")
    exit(1)

print(f"\n--- 총 {len(records_to_upsert)}건의 항공사 정보 업데이트/추가 시작 ({csv_file_path} 파일로부터) ---")

updated_count = 0
inserted_count = 0

for record in records_to_upsert:
    airline_code = record.get('airline_code')
    # airline_name_from_schedule_api 필드를 airline_name_kor 필드에 매핑합니다.
    airline_name_kor = record.get('airline_name_kor')
    airline_contact = record.get('airline_contact')

    if not airline_name_kor:
        print(f"⚠️ 경고: 항공사 이름이 없어 건너뜁니다: {record}")
        continue

    # 업데이트 또는 삽입할 문서 내용
    update_document = {
        'airline_name_kor': airline_name_kor,
        'airline_code': airline_code,
        'airline_contact': airline_contact
    }
    
    # 쿼리 필터: 항공사 한글명으로 기존 문서를 찾습니다.
    # 이름 불일치 방지를 위해 .strip()으로 공백 제거
    filter_query = {'airline_name_kor': airline_name_kor.strip()}

    # update_one 메서드를 upsert=True 옵션과 함께 사용합니다.
    result = airline_collection.update_one(
        filter_query,
        {'$set': update_document},
        upsert=True # 문서가 없으면 새로 삽입합니다.
    )

    if result.upserted_id:
        inserted_count += 1
        print(f"✅ 새 항공사 추가: '{airline_name_kor}' (IATA: {airline_code or '없음'})")
    elif result.modified_count > 0:
        updated_count += 1
        print(f"⬆️ 항공사 정보 업데이트: '{airline_name_kor}' (IATA: {airline_code or '없음'})")
    else:
        # 변경사항이 없지만, 이미 존재하고 일치하는 경우
        print(f"➖ 항공사 정보 변화 없음: '{airline_name_kor}'")


print(f"\n--- 항공사 정보 업데이트/추가 완료 ---")
print(f"총 {inserted_count}건의 새 항공사 정보가 추가되었습니다.")
print(f"총 {updated_count}건의 기존 항공사 정보가 업데이트되었습니다.")
print(f"총 {len(records_to_upsert) - inserted_count - updated_count}건의 항공사 정보는 변경사항이 없었습니다.")


# --- MongoDB 연결 종료 ---
client.close()
print("\n--- MongoDB 연결 종료 ---")