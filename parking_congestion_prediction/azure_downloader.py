import os
import pandas as pd
from azure.storage.blob import BlobServiceClient
from azure.identity import DefaultAzureCredential
from dotenv import load_dotenv
import io

load_dotenv()

# 환경 변수 설정
AZURE_STORAGE_ACCOUNT_NAME = os.getenv("AZURE_STORAGE_ACCOUNT_NAME")
CONTAINER_NAME = "datacollector"

# 최종 병합된 파일들이 저장될 로컬 경로
OUTPUT_FOLDER = "./merged_data"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

if not AZURE_STORAGE_ACCOUNT_NAME:
    raise ValueError("AZURE_STORAGE_ACCOUNT_NAME 환경 변수가 설정되지 않았습니다.")

try:
    credential = DefaultAzureCredential()
    blob_service_client = BlobServiceClient(
        account_url=f"https://{AZURE_STORAGE_ACCOUNT_NAME}.blob.core.windows.net",
        credential=credential
    )
    container_client = blob_service_client.get_container_client(CONTAINER_NAME)

    print(f"📦 Processing blobs from container '{CONTAINER_NAME}'...")

    flight_dataframes = []
    parking_dataframes = []
    
    processed_count = 0
    flight_files_count = 0
    parking_files_count = 0

    for blob in container_client.list_blobs():
        full_blob_path = blob.name # Blob의 전체 경로 또는 이름
        # 여기서 blob의 전체 경로에서 파일명만 추출
        blob_filename = os.path.basename(full_blob_path) # <-- 이 부분이 핵심 수정

        print(f"Reading: {full_blob_path} (Filename for check: {blob_filename})")

        try:
            # 파일 이름에 따라 분류 (이제 blob_filename으로 검사)
            if blob_filename.startswith("flight_data_") and blob_filename.endswith(".csv"):
                target_list = flight_dataframes
                file_type = "Flight Data"
                flight_files_count += 1
            elif blob_filename.startswith("parking_data_") and blob_filename.endswith(".csv"):
                target_list = parking_dataframes
                file_type = "Parking Data"
                parking_files_count += 1
            else:
                print(f"Skipping unknown file type or format mismatch: {full_blob_path}")
                continue

            blob_data = container_client.download_blob(full_blob_path).readall() # 다운로드는 full_blob_path로
            data_stream = io.BytesIO(blob_data)
            
            df = pd.read_csv(data_stream, encoding='utf-8', low_memory=False)
            
            target_list.append(df)
            processed_count += 1
            print(f"✅ Processed {file_type}: {full_blob_path}")

        except Exception as file_ex:
            print(f"⚠️ Error processing {full_blob_path}: {file_ex}")
            continue

    print(f"\n🎉 Total {processed_count} files processed. ({flight_files_count} flight files, {parking_files_count} parking files)")

    # ------------------- 항공편 데이터 병합 및 저장 -------------------
    if flight_dataframes:
        print("\nMerging flight dataframes...")
        merged_flight_df = pd.concat(flight_dataframes, ignore_index=True)
        print(f"Total rows in merged flight data: {len(merged_flight_df)}")

        output_flight_file_path = os.path.join(OUTPUT_FOLDER, "merged_flight_data.csv")
        merged_flight_df.to_csv(output_flight_file_path, index=False, encoding='utf-8-sig')
        print(f"✅ Merged flight data saved to: {output_flight_file_path}")
    else:
        print("No flight data files were processed or merged.")

    # ------------------- 주차장 데이터 병합 및 저장 -------------------
    if parking_dataframes:
        print("\nMerging parking dataframes...")
        merged_parking_df = pd.concat(parking_dataframes, ignore_index=True)
        print(f"Total rows in merged parking data: {len(merged_parking_df)}")

        output_parking_file_path = os.path.join(OUTPUT_FOLDER, "merged_parking_data.csv")
        merged_parking_df.to_csv(output_parking_file_path, index=False, encoding='utf-8-sig')
        print(f"✅ Merged parking data saved to: {output_parking_file_path}")
    else:
        print("No parking data files were processed or merged.")

except Exception as ex:
    print(f"\n🚨 Critical Error during Blob Service operation: {ex}")
    print("다음 사항을 확인하세요:")
    print("1. Azure CLI에 'az login'으로 로그인되어 있는지 확인하세요.")
    print("2. 'AZURE_STORAGE_ACCOUNT_NAME' 환경 변수가 올바르게 설정되어 있는지 확인하세요.")
    print("3. 현재 Azure 계정에 해당 스토리지 계정 및 컨테이너에 대한 'Storage Blob 데이터 Reader' 역할이 부여되어 있는지 확인하세요.")
    print("4. 컨테이너 이름이 올바른지 확인하세요 (datacollector).")