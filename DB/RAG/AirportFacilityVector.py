print("스크립트 시작!") # 코드 최상단에 추가
from pymongo import MongoClient
import os
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
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
    airport_collection_1 = db["AirportEnterprise"]
    airport_collection_2 = db["AirportFacility"]
    airport_vectors_collection = db["AirportFacilityVector"]

    print("MongoDB에 성공적으로 연결되었습니다.")
    print(f"원본 컬렉션: {airport_collection_1.name}, {airport_collection_2.name}")
    print(f"대상 컬렉션: {airport_vectors_collection.name}")

    # 이전 실행으로 남아있는 데이터가 있다면 삭제
    user_input = input(f"'{airport_vectors_collection.name}' 컬렉션의 기존 데이터를 모두 삭제하시겠습니까? (y/n): ")
    if user_input.lower() == 'y':
        result = airport_vectors_collection.delete_many({})
        print(f"기존 {result.deleted_count}개 문서 삭제 완료.")

    print("\n'Airport' 컬렉션에서 문서를 불러와 임베딩 및 저장 중...")


    # ----- AirportEnterprise -----
    # 모든 문서 불러오기 (커서 사용)
    documents_cursor = airport_collection_1.find({})

    # 총 문서 개수를 미리 알아내어 tqdm으로 진행률 표시
    total_documents = airport_collection_1.count_documents({})
    
    processed_documents_count = 0
    documents_to_insert = []


    for doc in tqdm(documents_cursor, total=total_documents, desc="Processing"):
        try:
            # 3. 임베딩할 텍스트 구성
            enterprise_name = doc.get('enterprise_name', '')
            item = doc.get('item', '')
            location = doc.get('location', '')
            service_time = doc.get('service_time', '')
            arrordep = doc.get('arrordep', '')
            if arrordep == 'A':
                arrordep = '입국장'
            elif arrordep == 'D':
                arrordep = '출국장'
            tel = doc.get('tel', '')

            # 의미를 잘 전달할 수 있는 문장 구성
            text_to_embed = f"{enterprise_name} 시설은 {item}를 취급합니다. 위치는 {location}입니다. 운영시간은 {service_time}입니다. {arrordep}구역입니다. 전화번호는 {tel}입니다."

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
    
    # ----- AirportFacility -----
    # 모든 문서 불러오기 (커서 사용)
    documents_cursor_2 = airport_collection_2.find({})

    # 총 문서 개수를 미리 알아내어 tqdm으로 진행률 표시
    total_documents_2 = airport_collection_2.count_documents({})
    
    processed_documents_count = 0
    documents_to_insert = []


    for doc in tqdm(documents_cursor_2, total=total_documents_2, desc="Processing"):
        try:
            # 3. 임베딩할 텍스트 구성
            facility_id = doc.get('facility_id', '')
            
            arrordep = doc.get('arrordep', '')
            if arrordep == 'A':
                arrordep = '입국장'
            elif arrordep == 'D':
                arrordep = '출국장'

            facility_name = doc.get('facility_name', '')
            
            floor_info = doc.get('floor_info', '')
                        
            is_duty_free_location = doc.get('is_duty_free_location', '')
            if is_duty_free_location == 'Y':
                is_duty_free_location = '면세지역'
            elif is_duty_free_location == 'N':
                is_duty_free_location = '비면세지역'

            item = doc.get('item', '')
            
            large_category = doc.get('large_category', '')
            medium_category = doc.get('medium_category', '')
            small_category = doc.get('small_category', '')
            
            location = doc.get('location', '')
            
            terminal_id = doc.get('terminal_id', '')
            if terminal_id == 'P01':
                terminal_id = '제 1터미널'
            elif terminal_id == 'P03':
                terminal_id = '제 2터미널'
            elif terminal_id == 'G01':
                terminal_id = '탑승동'
            elif terminal_id == 'G02':
                terminal_id = '제 1터미널 교통센터'
            elif terminal_id == 'G03':
                terminal_id = '제 2터미널 교통센터'
                                
            service_time = doc.get('service_time', '')
            tel = doc.get('tel', '')

            # 의미를 잘 전달할 수 있는 문장 구성
            text_to_embed = f"{facility_id}번 시설명은 {facility_name}입니다. {item}품목을 취급합니다. 대분류는 {large_category}, 중분류는 {medium_category}, 소분류는 {small_category}입니다. {is_duty_free_location}에 위치합니다. {arrordep}구역입니다. {terminal_id}지역입니다. {floor_info}층입니다. 정확한 위치는 {location}입니다. 운영시간은 {service_time}입니다. 시설 전화번호는 {tel}입니다."

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