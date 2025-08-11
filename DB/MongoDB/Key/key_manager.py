from pymongo import MongoClient
import os
from dotenv import load_dotenv
import requests
from requests import Request
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
                text_data = response.text.strip()

                # 1) JSON 응답일 경우
                if text_data.startswith("{") or text_data.startswith("["):
                    try:
                        data = response.json()
                        result_code = data.get("response", {}).get("header", {}).get("resultCode")
                        if result_code == "00":
                            print("키 불러오기 성공 (JSON 정상)")
                            return key
                        elif data.get("currentCount") is not None:
                            print("키 불러오기 성공 (currentCount 발견)")
                            return key
                        else:
                            print(f"[경고] JSON 응답 코드 이상: {result_code}")
                    except json.JSONDecodeError:
                        print("[경고] JSON 파싱 실패 (JSON 포맷 아님)")

                # 2) 기상청 텍스트 응답일 경우
                elif "#START7777" in text_data and "#7777END" in text_data:
                    print("키 불러오기 성공 (#START7777 ~ #7777END 패턴 발견)")
                    return key

                # 3) 예상치 못한 형식
                else:
                    print(f"[경고] 예상치 못한 응답 형식. 응답 일부: {text_data[:100]}")

            else:
                print(f"[경고] 요청 실패 - 상태코드: {response.status_code}")

        except Exception as e:
            print(f"[예외] 키 {i+1} 오류 발생: {e}")

        time.sleep(2)

    print("유효한 API 키를 찾지 못했습니다.")
    return None
