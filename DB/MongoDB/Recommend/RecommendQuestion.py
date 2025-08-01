from pymongo import MongoClient
import time
from datetime import datetime
from dotenv import load_dotenv
import os
import pandas as pd
from pymongo import MongoClient
from dotenv import load_dotenv 
import os 

load_dotenv()  # .env 파일에서 환경변수 불러오기
mongo_uri = os.getenv("MONGO_URI")

# MongoDB 설정
client = MongoClient(mongo_uri)
db = client["AirBot"]
collection = db["AirportCongestionPredict"]

EXCEL_FILES_DIR = ".\DB\MongoDB\Recommend/" 

visa_file = os.path.join(EXCEL_FILES_DIR, "recommend_question_data.csv")

def upload_recommend_question_data(file_path, db):
    collection_name = "RecommendQuestion"
    try:
        df = pd.read_csv(file_path)
        
        df.rename(columns={
            'intent': 'intent',
            'recommend_question': 'recommend_question'
        }, inplace=True)
        
        string_cols = ['intent', 'recommend_question']
        
        for col in string_cols:
            if col in df.columns:
                df[col] = df[col].apply(lambda x: None if pd.isna(x) else str(x))
            else:
                df[col] = None
                
        data = df.to_dict(orient="records")
        
        collection = db[collection_name]
        
        collection.delete_many({}) # 기존 데이터 삭제
        collection.insert_many(data)
        print(f"✅ 성공적으로 '{file_path}' 데이터를 '{collection_name}' 컬렉션에 삽입했습니다.")
    except FileNotFoundError:
        print(f"❌ 오류: 파일을 찾을 수 없습니다. 경로를 확인해주세요: {file_path}")
    except Exception as e:
        print(f"❌ '{file_path}' 처리 중 오류가 발생했습니다: {e}")

# --- 실행 ---
upload_recommend_question_data(visa_file, db)

client.close()
print("\n--- 모든 데이터 업로드 완료 및 MongoDB 연결 종료 ---")
      