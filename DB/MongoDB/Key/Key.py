import json
from pymongo import MongoClient
from datetime import datetime
from dotenv import load_dotenv
import os

load_dotenv()  # .env 파일에서 환경변수 불러오기
mongo_uri = os.getenv("MONGO_URI")

# MongoDB 설정
client = MongoClient(mongo_uri)
db = client["AirBot"]
collection = db["Key"]

def insert_api_keys_to_mongo(json_file_path):
    # JSON 파일에서 키 데이터 읽기
    with open(json_file_path, 'r') as file:
        key_data_list = json.load(file)

    for key_data in key_data_list:
        doc = {
            "type": key_data["type"],
            "content": key_data["content"],
            "is_valid": True,
        }
        # 중복 방지: 이미 존재하는 키인지 확인 후 없으면 삽입
        if not collection.find_one({"type": doc["type"], "content": doc["content"]}):
            collection.insert_one(doc)

if __name__ == "__main__":
    json_path = "C:/Users/User/Desktop/Aivle/빅프로젝트/BigProject/MongoDB/key.json"
    insert_api_keys_to_mongo(json_path)