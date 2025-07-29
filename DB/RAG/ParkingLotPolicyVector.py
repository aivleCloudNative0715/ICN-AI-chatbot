print("스크립트 시작!") # 코드 최상단에 추가
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
import math
from tqdm import tqdm # 진행률 표시를 위한 라이브러리

# .env 파일에서 환경변수 불러오기 (MONGO_URI)
load_dotenv()
mongo_uri = os.getenv("MONGO_URI")

# 1. 임베딩 모델 로드
print("임베딩 모델을 로드 중입니다... (최초 실행 시 다운로드로 인해 시간이 소요될 수 있습니다.)")
try:
    embedding_model = SentenceTransformer('dragonkue/snowflake-arctic-embed-l-v2.0-ko')
    print("임베딩 모델 로드 완료.")
except Exception as e:
    print(f"임베딩 모델 로드 중 오류 발생: {e}")
    print("pip install sentence-transformers 를 실행했는지 확인해주세요.")
    exit()

# 2. MongoDB 클라이언트 연결
client = None 
try:
    client = MongoClient(mongo_uri)
    db = client["AirBot"]
    airport_collection_1 = db["ParkingFeeDiscountPolicy"]
    airport_collection_2 = db["ParkingFeePolicy"]
    airport_collection_3 = db["ParkingFeePayment"]
    airport_vectors_collection = db["ParkingLotPolicyVector"]

    print("MongoDB에 성공적으로 연결되었습니다.")
    print(f"원본 컬렉션: {airport_collection_1.name}, {airport_collection_2.name}, {airport_collection_3.name}")
    print(f"대상 컬렉션: {airport_vectors_collection.name}")

    # 이전 실행으로 남아있는 데이터가 있다면 삭제
    user_input = input(f"'{airport_vectors_collection.name}' 컬렉션의 기존 데이터를 모두 삭제하시겠습니까? (y/n): ")
    if user_input.lower() == 'y':
        result = airport_vectors_collection.delete_many({})
        print(f"기존 {result.deleted_count}개 문서 삭제 완료.")

    print("\n컬렉션에서 문서를 불러와 임베딩 및 저장 중...")


    # 1. ----- ParkingFeeDiscountPolicy -----
    # 모든 문서 불러오기 (커서 사용)
    documents_cursor = airport_collection_1.find({})

    # 총 문서 개수를 미리 알아내어 tqdm으로 진행률 표시
    total_documents = airport_collection_1.count_documents({})
    
    processed_documents_count = 0
    documents_to_insert = []


    for doc in tqdm(documents_cursor, total=total_documents, desc="Processing"):
        try:
            # 3. 임베딩할 텍스트 구성
            discount_policy_title = doc.get('discount_policy_title', '')      
            discount_condition = doc.get('discount_condition', '')
   
            discount_rate = doc.get('discount_rate', '') * 100
            notice = doc.get('notice', '')
            post_submission_discount_document = doc.get('post_submission_discount_document', '')
            realtime_discount_document = doc.get('realtime_discount_document', '')

            # 의미를 잘 전달할 수 있는 문장 구성
            text_to_embed = ''
            if discount_policy_title == "다자녀가구":
                text_to_embed = f"{discount_condition}에 속하는 {discount_policy_title}는 {discount_rate}%의 주차 요금 감면을 받을 수 있습니다. 단 {realtime_discount_document}입니다. 사후 감면을 받기 위해서는, 인정을 위해 제출해야 할 서류로 차량 등록증(부/모/직계존속 명의), 주민등록등본이 있습니다. 단 2자녀 이상 막내 만 18세 이하 대상 부모 또는 부모의 직계 존속만 사후 환불 홈페이지에서 사후 감면 신청이 가능합니다. {notice}"
            else:
                text_to_embed=f'{discount_condition}에 속하는 {discount_policy_title}는 {discount_rate}%의 주차 요금 감면을 받을 수 있습니다. 행정정보 조회로 차량확인이 가능한 경우에만 실시간 할인을 받을 수 있습니다. 이 경우 그 자리에서 제출할 서류는 별도로 없습니다. 사후 감면을 받기 위해서는 {post_submission_discount_document}를 제출해야 합니다. {notice}'
                

            if not text_to_embed.strip(): # 텍스트가 비어있으면 건너뛰기
                print(f"경고: _id {doc.get('_id')} 문서에서 임베딩할 텍스트를 생성할 수 없습니다. 건너_id")
                continue

            # 4. 임베딩 생성
            embedding = embedding_model.encode(text_to_embed).tolist() # NumPy 배열을 Python 리스트로 변환

            # 5. 새로운 문서 생성
            new_doc = {
                "original_id": doc['_id'], # 원본 문서의 _id를 참조용으로 저장
                "text_content": text_to_embed, # 임베딩에 사용된 원본 텍스트도 저장 (나중에 RAG에서 사용)
                "embedding": embedding # 생성된 벡터 임베딩 저장 필드
            }
            documents_to_insert.append(new_doc)
            processed_documents_count += 1

            # 6. 1000개마다 배치 삽입
            if len(documents_to_insert) >= 1000:
                airport_vectors_collection.insert_many(documents_to_insert)
                documents_to_insert = [] # 리스트 초기화

        except Exception as doc_error:
            print(f"\n문서 처리 중 오류 발생 (ID: {doc.get('_id')}): {doc_error}")
            continue # 다음 문서로 계속 진행

    # 남은 문서들 일괄 삽입
    if documents_to_insert:
        airport_vectors_collection.insert_many(documents_to_insert)

    print(f"\n총 {processed_documents_count}개의 문서를 임베딩하여 '{airport_vectors_collection.name}' 컬렉션에 저장했습니다.")
    
    
    # 2. ----- ParkingFeePolicy -----
    # 모든 문서 불러오기 (커서 사용)
    documents_cursor_2 = airport_collection_2.find({})

    # 총 문서 개수를 미리 알아내어 tqdm으로 진행률 표시
    total_documents_2 = airport_collection_2.count_documents({})
    
    processed_documents_count = 0
    documents_to_insert = []


    for doc in tqdm(documents_cursor_2, total=total_documents_2, desc="Processing"):
        try:
            # 3. 임베딩할 텍스트 구성
            policy_title = doc.get('policy_title', '')
            daily_max_price_krw = doc.get('daily_max_price_krw', '')
            extra_unit_duration_minutes = doc.get('extra_unit_duration_minutes', '')
            extra_unit_price_krw = doc.get('extra_unit_price_krw', '')
            inital_dueation_minutes = doc.get('inital_dueation_minutes', '')
            initial_price_krw = doc.get('initial_price_krw', '')
            is_free = doc.get('is_free', '')

            # 의미를 잘 전달할 수 있는 문장 구성
            text_to_embed = ''
            if is_free == True:
                text_to_embed = f"{policy_title}에 해당할 경우 주차 요금은 무료입니다"
                
            elif math.isnan(initial_price_krw) or initial_price_krw == 0:
                if daily_max_price_krw == 0:             
                    text_to_embed = f"{policy_title}에 해당할 경우 추가 {extra_unit_duration_minutes}분 당 {extra_unit_price_krw}원의 주차요금이 부과됩니다."
                else:
                    text_to_embed = f"{policy_title}에 해당할 경우 추가 {extra_unit_duration_minutes}분 당 {extra_unit_price_krw}원의 주차요금이 부과됩니다. 일 최대 {daily_max_price_krw}원까지 부과됩니다."
            elif daily_max_price_krw == 0:
                text_to_embed = f"{policy_title}에 해당할 경우 최초 {inital_dueation_minutes}분에 한해 {initial_price_krw}원의 주차요금이 부과됩니다. 추가 {extra_unit_duration_minutes}분 당 {extra_unit_price_krw}원의 주차요금이 부과됩니다."
            else:
                text_to_embed = f"{policy_title}에 해당할 경우 최초 {inital_dueation_minutes}분에 한해 {initial_price_krw}원의 주차요금이 부과됩니다. 추가 {extra_unit_duration_minutes}분 당 {extra_unit_price_krw}원의 주차요금이 부과됩니다. 일 최대 {daily_max_price_krw}원까지 부과됩니다." 
                
            if not text_to_embed.strip(): # 텍스트가 비어있으면 건너뛰기
                print(f"경고: _id {doc.get('_id')} 문서에서 임베딩할 텍스트를 생성할 수 없습니다. 건너_id")
                continue

            # 4. 임베딩 생성
            embedding = embedding_model.encode(text_to_embed).tolist() # NumPy 배열을 Python 리스트로 변환

            # 5. 새로운 문서 생성
            new_doc = {
                "original_id": doc['_id'], # 원본 문서의 _id를 참조용으로 저장
                "text_content": text_to_embed, # 임베딩에 사용된 원본 텍스트도 저장 (나중에 RAG에서 사용)
                "embedding": embedding # 생성된 벡터 임베딩 저장 필드
            }
            documents_to_insert.append(new_doc)
            processed_documents_count += 1

            # 6. 1000개마다 배치 삽입
            if len(documents_to_insert) >= 1000:
                airport_vectors_collection.insert_many(documents_to_insert)
                documents_to_insert = [] # 리스트 초기화

        except Exception as doc_error:
            print(f"\n문서 처리 중 오류 발생 (ID: {doc.get('_id')}): {doc_error}")
            continue # 다음 문서로 계속 진행

    # 남은 문서들 일괄 삽입
    if documents_to_insert:
        airport_vectors_collection.insert_many(documents_to_insert)

    print(f"\n총 {processed_documents_count}개의 문서를 임베딩하여 '{airport_vectors_collection.name}' 컬렉션에 저장했습니다.")

    # 3. ----- ParkingFeePayment -----
    # 모든 문서 불러오기 (커서 사용)
    documents_cursor_3 = airport_collection_3.find({})

    # 총 문서 개수를 미리 알아내어 tqdm으로 진행률 표시
    total_documents_3 = airport_collection_3.count_documents({})
    
    processed_documents_count = 0
    documents_to_insert = []


    for doc in tqdm(documents_cursor_3, total=total_documents_3, desc="Processing"):
        try:
            # 3. 임베딩할 텍스트 구성
            payment_title = doc.get('payment_title', '')
                     
            available_cash = doc.get('available_cash', '')
            if available_cash:
                available_cash = "가능"
            else:
                available_cash = "불가능" 
                
            available_credit = doc.get('available_credit', '')
            if available_credit:
                available_credit = "가능"
            else:
                available_credit = "불가능" 
                
            available_hipass = doc.get('available_hipass', '')
            if available_hipass:
                available_hipass = "가능"
            else:
                available_hipass = "불가능" 
                
            available_postpaid = doc.get('available_postpaid', '')
            if available_postpaid:
                available_postpaid = "가능"
            else:
                available_postpaid = "불가능" 
                
            available_prepaid = doc.get('available_prepaid', '')
            if available_prepaid:
                available_prepaid = "가능"
            else:
                available_prepaid = "불가능" 
                
            available_transit = doc.get('available_transit', '')
            if available_transit:
                available_transit = "가능"
            else:
                available_transit = "불가능"
                
            payment_step_description = doc.get('payment_step_description', '')
            

            # 의미를 잘 전달할 수 있는 문장 구성
            text_to_embed = ''
            text_to_embed = f"주차요금 정산 방법에 대한 설명입니다. {payment_title}는 현금 {available_cash}, 신용카드 {available_credit}, 하이패스 {available_hipass}, 사후 지불 {available_postpaid}, 사전 지불 {available_prepaid}, 경유 {available_transit}. 절차 안내: {payment_step_description}"

            if not text_to_embed.strip(): # 텍스트가 비어있으면 건너뛰기
                print(f"경고: _id {doc.get('_id')} 문서에서 임베딩할 텍스트를 생성할 수 없습니다. 건너_id")
                continue

            # 4. 임베딩 생성
            embedding = embedding_model.encode(text_to_embed).tolist() # NumPy 배열을 Python 리스트로 변환

            # 5. 새로운 문서 생성
            new_doc = {
                "original_id": doc['_id'], # 원본 문서의 _id를 참조용으로 저장
                "text_content": text_to_embed, # 임베딩에 사용된 원본 텍스트도 저장 (나중에 RAG에서 사용)
                "embedding": embedding # 생성된 벡터 임베딩 저장 필드
            }
            documents_to_insert.append(new_doc)
            processed_documents_count += 1

            # 6. 1000개마다 배치 삽입
            if len(documents_to_insert) >= 1000:
                airport_vectors_collection.insert_many(documents_to_insert)
                documents_to_insert = [] # 리스트 초기화

        except Exception as doc_error:
            print(f"\n문서 처리 중 오류 발생 (ID: {doc.get('_id')}): {doc_error}")
            continue # 다음 문서로 계속 진행

    # 남은 문서들 일괄 삽입
    if documents_to_insert:
        airport_vectors_collection.insert_many(documents_to_insert)

    print(f"\n총 {processed_documents_count}개의 문서를 임베딩하여 '{airport_vectors_collection.name}' 컬렉션에 저장했습니다.")


except Exception as e:
    print(f"MongoDB 작업 중 치명적인 오류 발생: {e}")

finally:
    if client:
        client.close()
        print("\nMongoDB 연결이 종료되었습니다.")