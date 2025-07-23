from pymongo import MongoClient
import os
from dotenv import load_dotenv
import requests
import time
import json

load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

client = MongoClient(mongo_uri)
db = client["AirBot"]
collection = db["Key"]

def get_valid_api_key(url, params_base, key_type, auth_param_name, max_retries=5):
    keys = list(collection.find({"type": key_type, "is_valid": True}))
    print(f"조회된 키 개수: {len(keys)}")

    for i, key_doc in enumerate(keys[:max_retries]):
        key = key_doc["content"].strip()
        key_id = key_doc["_id"]
        print(f"테스트 중인 키 {i+1}: {key}")

        params = params_base.copy()
        params[auth_param_name] = key

        try:
            response = requests.get(url, params=params, timeout=10)
            if response.status_code == 200:
                try:
                    data = response.json()
                    result_code = data.get("response", {}).get("header", {}).get("resultCode")
                    if result_code == "00":
                        print("키 불러오기 성공 (JSON 정상)")
                        return key
                    else:
                        print(f"[경고] 키 {i+1} 응답 코드: {result_code}")
                except json.JSONDecodeError:
                    # JSON 파싱 실패 시 텍스트에 특정 문자열 포함 여부 확인
                    if "#START7777" in response.text:
                        print("키 불러오기 성공 (#START7777 발견)")
                        return key
                    else:
                        print(f"[경고] 키 {i+1} 응답이 JSON이 아니고 #START7777도 없음. 응답 일부: {response.text[:100]}")
            else:
                print(f"[경고] 키 {i+1} 요청 실패 - 상태코드: {response.status_code}")
        except Exception as e:
            print(f"[예외] 키 {i+1} 오류 발생: {e}")

        # 실패 처리
        collection.update_one({"_id": key_id}, {"$set": {"is_valid": False}})
        time.sleep(2)

    print("유효한 API 키를 찾지 못했습니다.")
    return None
