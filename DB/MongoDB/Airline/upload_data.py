import pandas as pd
from pymongo import MongoClient
from pymongo.server_api import ServerApi
from dotenv import load_dotenv
import os

load_dotenv()

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

visa_file = os.path.join(BASE_DIR, "visa.xlsx")
restricted_item_file = os.path.join(BASE_DIR, "restricted_item.xlsx")
min_transit_time_file = os.path.join(BASE_DIR, "최소_환승_시간.xlsx")
airport_procedure_file = os.path.join(BASE_DIR, "공항_절차.xlsx")
transit_path_file = os.path.join(BASE_DIR, "환승_경로.xlsx")


def _get_mongo_client_and_db():
    mongo_uri = os.getenv("MONGO_URI")
    if not mongo_uri:
        raise ValueError("오류: .env 파일에서 MONGO_URI를 찾을 수 없습니다.")
    try:
        client = MongoClient(mongo_uri, server_api=ServerApi('1'))
        db = client["AirBot"]
        client.admin.command('ping')
        return client, db
    except Exception as e:
        raise ConnectionError(f"MongoDB 연결 오류: {e}")

def _upload_with_temp_collection(file_path, collection_name, rename_map=None, process_df_func=None):
    client, db = _get_mongo_client_and_db()
    temp_collection_name = collection_name + "_temp"
    try:
        df = pd.read_excel(file_path)
        if rename_map:
            df.rename(columns=rename_map, inplace=True)
        if process_df_func:
            df = process_df_func(df)
        else:
            for col in df.columns:
                df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x) if isinstance(x, str) else x)

        data = df.to_dict(orient="records")

        temp_col = db[temp_collection_name]
        temp_col.delete_many({})
        temp_col.insert_many(data)

        if collection_name in db.list_collection_names():
            db.drop_collection(collection_name)

        db[temp_collection_name].rename(collection_name)

        print(f"'{file_path}' -> '{collection_name}' 컬렉션 업데이트 완료 (임시 컬렉션 스왑)")

    except FileNotFoundError:
        print(f"오류: 파일을 찾을 수 없습니다: {file_path}")
    except Exception as e:
        print(f"'{file_path}' 처리 중 오류 발생: {e}")
    finally:
        client.close()


def upload_country_data():
    rename_map = {
        'country_c': 'country_code',
        'country_n.': 'country_name_kor',
        'visa_requi': 'visa_required',
        'stay_durat': 'stay_duration',
        'entry_requ': 'entry_requirement'
    }
    def process(df):
        df['visa_required'] = df['visa_required'].astype(bool)
        df['stay_duration'] = pd.to_numeric(df['stay_duration'], errors='coerce').apply(lambda x: None if pd.isna(x) else int(x))
        df['entry_requirement'] = df['entry_requirement'].apply(lambda x: None if pd.isna(x) else str(x))
        return df

    _upload_with_temp_collection(visa_file, "Country", rename_map, process)

def upload_restricted_item_data():
    rename_map = {
        'item_category': 'item_category',
        'item_name': 'item_name',
        'carry_on_policy': 'carry_on_policy',
        'checked_baggage_policy': 'checked_baggage_policy',
        'source_url': 'source_url'
    }
    _upload_with_temp_collection(restricted_item_file, "RestrictedItem", rename_map)

def upload_minimum_connection_time_data():
    rename_map = {
        'origin_terminal': 'origin_terminal',
        'destination_terminal': 'destination_terminal',
        'min_time_minutes': 'min_time_minutes'
    }
    def process(df):
        df['min_time_minutes'] = pd.to_numeric(df['min_time_minutes'], errors='coerce').apply(lambda x: None if pd.isna(x) else int(x))
        df['origin_terminal'] = df['origin_terminal'].apply(lambda x: None if pd.isna(x) else str(x))
        df['destination_terminal'] = df['destination_terminal'].apply(lambda x: None if pd.isna(x) else str(x))
        return df

    _upload_with_temp_collection(min_transit_time_file, "MinimumConnectionTime", rename_map, process)

def upload_airport_procedure_data():
    rename_map = {
        'procedure_type': 'procedure_type',
        'step_number': 'step_number',
        'sub_step': 'sub_step',
        'step_name': 'step_name',
        'description': 'description',
        'source_url': 'source_url'
    }
    def process(df):
        df['step_number'] = pd.to_numeric(df['step_number'], errors='coerce').apply(lambda x: None if pd.isna(x) else int(x))
        if 'sub_step' in df.columns:
            df['sub_step'] = pd.to_numeric(df['sub_step'], errors='coerce').apply(lambda x: None if pd.isna(x) else int(x))
        else:
            df['sub_step'] = None
        for col in ['procedure_type', 'step_name', 'description', 'source_url']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x))
            else:
                df[col] = None
        return df

    _upload_with_temp_collection(airport_procedure_file, "AirportProcedure", rename_map, process)

def upload_transit_path_data():
    rename_map = {
        'origin_terminal': 'origin_terminal',
        'destination_terminal': 'destination_terminal',
        'path_description': 'path_description'
    }
    def process(df):
        for col in ['origin_terminal', 'destination_terminal', 'path_description']:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x))
            else:
                df[col] = None
        return df

    _upload_with_temp_collection(transit_path_file, "TransitPath", rename_map, process)


if __name__ == "__main__":
    upload_country_data()
    upload_restricted_item_data()
    upload_minimum_connection_time_data()
    upload_airport_procedure_data()
    upload_transit_path_data()
