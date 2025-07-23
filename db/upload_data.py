import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv 
import os 


load_dotenv()

mongo_uri = os.getenv("MONGO_URI")

if not mongo_uri:
    print("❌ 오류: .env 파일에서 MONGO_URI를 찾을 수 없습니다. 파일을 확인해주세요.")
    exit(1) # 프로그램 종료

# --- MongoDB 연결 설정 ---
try:
    client = MongoClient(mongo_uri, server_api=ServerApi('1'))
    db = client["AirBot"] 

    client.admin.command('ping')
    print("✅ MongoDB에 성공적으로 연결되었습니다!")

except Exception as e:
    print(f"❌ MongoDB 연결 오류: {e}")
    exit(1) # 연결 실패 시 프로그램 종료


EXCEL_FILES_DIR = "./" 

visa_file = os.path.join(EXCEL_FILES_DIR, "visa.xlsx")
restricted_item_file = os.path.join(EXCEL_FILES_DIR, "restricted_item.xlsx")
min_transit_time_file = os.path.join(EXCEL_FILES_DIR, "최소 환승 시간.xlsx")
airport_procedure_file = os.path.join(EXCEL_FILES_DIR, "공항 절차.xlsx")
transit_path_file = os.path.join(EXCEL_FILES_DIR, "환승 경로.xlsx") 


# --- 각 컬렉션별 데이터 업로드 함수 정의 ---

def upload_country_data(file_path, db):
    collection_name = "Country"
    try:
        df = pd.read_excel(file_path)
        df.rename(columns={
            'country_c': 'country_code',
            'country_n.': 'country_name_kor',
            'visa_requi': 'visa_required',
            'stay_durat': 'stay_duration',
            'entry_requ': 'entry_requirement'
        }, inplace=True)
        df['visa_required'] = df['visa_required'].astype(bool)
        df['stay_duration'] = pd.to_numeric(df['stay_duration'], errors='coerce')
        df['stay_duration'] = df['stay_duration'].apply(lambda x: None if pd.isna(x) else int(x))
        df['entry_requirement'] = df['entry_requirement'].apply(lambda x: None if pd.isna(x) else str(x))
        data = df.to_dict(orient="records")
        collection = db[collection_name]
        # collection.delete_many({}) # 기존 데이터 삭제를 원하면 주석 해제
        collection.insert_many(data)
        print(f"✅ 성공적으로 '{file_path}' 데이터를 '{collection_name}' 컬렉션에 삽입했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
    except Exception as e:
        print(f"❌ '{file_path}' 처리 중 오류가 발생했습니다: {e}")

def upload_restricted_item_data(file_path, db):
    collection_name = "RestrictedItem"
    try:
        df = pd.read_excel(file_path)
        df.rename(columns={
            'item_category': 'item_category',
            'item_name': 'item_name',
            'carry_on_policy': 'carry_on_policy',
            'checked_baggage_policy': 'checked_baggage_policy',
            'source_url': 'source_url'
        }, inplace=True)
        for col in df.columns:
            df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x))
        data = df.to_dict(orient="records")
        collection = db[collection_name]
        # collection.delete_many({}) # 기존 데이터 삭제를 원하면 주석 해제
        collection.insert_many(data)
        print(f"✅ 성공적으로 '{file_path}' 데이터를 '{collection_name}' 컬렉션에 삽입했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
    except Exception as e:
        print(f"❌ '{file_path}' 처리 중 오류가 발생했습니다: {e}")

def upload_minimum_connection_time_data(file_path, db):
    collection_name = "MinimumConnectionTime"
    try:
        df = pd.read_excel(file_path)
        df.rename(columns={
            'origin_terminal': 'origin_terminal',
            'destination_terminal': 'destination_terminal',
            'min_time_minutes': 'min_time_minutes'
        }, inplace=True)
        df['min_time_minutes'] = pd.to_numeric(df['min_time_minutes'], errors='coerce')
        df['min_time_minutes'] = df['min_time_minutes'].apply(lambda x: None if pd.isna(x) else int(x))
        df['origin_terminal'] = df['origin_terminal'].apply(lambda x: None if pd.isna(x) else str(x))
        df['destination_terminal'] = df['destination_terminal'].apply(lambda x: None if pd.isna(x) else str(x))
        data = df.to_dict(orient="records")
        collection = db[collection_name]
        # collection.delete_many({}) # 기존 데이터 삭제를 원하면 주석 해제
        collection.insert_many(data)
        print(f"✅ 성공적으로 '{file_path}' 데이터를 '{collection_name}' 컬렉션에 삽입했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
    except Exception as e:
        print(f"❌ '{file_path}' 처리 중 오류가 발생했습니다: {e}")

def upload_airport_procedure_data(file_path, db):
    collection_name = "AirportProcedure"
    try:
        df = pd.read_excel(file_path)
        df.rename(columns={
            'procedure_type': 'procedure_type',
            'step_number': 'step_number',
            'sub_step': 'sub_step',
            'step_name': 'step_name',
            'description': 'description',
            'source_url': 'source_url'
        }, inplace=True)
        df['step_number'] = pd.to_numeric(df['step_number'], errors='coerce')
        df['step_number'] = df['step_number'].apply(lambda x: None if pd.isna(x) else int(x))
        if 'sub_step' in df.columns:
            df['sub_step'] = pd.to_numeric(df['sub_step'], errors='coerce')
            df['sub_step'] = df['sub_step'].apply(lambda x: None if pd.isna(x) else int(x))
        else:
            df['sub_step'] = None # 컬럼이 없으면 None으로 채움

        string_cols = ['procedure_type', 'step_name', 'description', 'source_url']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x))
            else:
                df[col] = None

        data = df.to_dict(orient="records")
        collection = db[collection_name]
        # collection.delete_many({}) # 기존 데이터 삭제를 원하면 주석 해제
        collection.insert_many(data)
        print(f"✅ 성공적으로 '{file_path}' 데이터를 '{collection_name}' 컬렉션에 삽입했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
    except Exception as e:
        print(f"❌ '{file_path}' 처리 중 오류가 발생했습니다: {e}")

def upload_transit_path_data(file_path, db):
    collection_name = "TransitPath"
    try:
        df = pd.read_excel(file_path)
        df.rename(columns={
            'origin_terminal': 'origin_terminal',
            'destination_terminal': 'destination_terminal',
            'path_description': 'path_description'
        }, inplace=True)
        string_cols = ['origin_terminal', 'destination_terminal', 'path_description']
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x))
            else:
                df[col] = None
        data = df.to_dict(orient="records")
        collection = db[collection_name]
        # collection.delete_many({}) # 기존 데이터 삭제를 원하면 주석 해제
        collection.insert_many(data)
        print(f"✅ 성공적으로 '{file_path}' 데이터를 '{collection_name}' 컬렉션에 삽입했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
    except Exception as e:
        print(f"❌ '{file_path}' 처리 중 오류가 발생했습니다: {e}")


# --- 모든 데이터 업로드 실행 ---

print("\n--- 모든 엑셀 데이터 MongoDB 업로드 시작 ---")

# 모든 컬렉션의 기존 데이터를 삭제하고 새로 넣고 싶다면,
# 각 함수 내부의 'collection.delete_many({})' 주석을 해제

upload_country_data(visa_file, db)
upload_restricted_item_data(restricted_item_file, db)
upload_minimum_connection_time_data(min_transit_time_file, db)
upload_airport_procedure_data(airport_procedure_file, db)
upload_transit_path_data(transit_path_file, db)

# --- MongoDB 연결 종료 ---
client.close()
print("\n--- 모든 데이터 업로드 완료 및 MongoDB 연결 종료 ---")