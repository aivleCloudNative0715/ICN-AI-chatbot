import time
import os
import logging
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

from MongoDB.Airline.airline import upload_airline_data_atomic
from MongoDB.Airline.airport import upload_airport_data_atomic
from MongoDB.Airline.flight_schedule import fetch_flight_schedule
from MongoDB.Airline.update_airline import update_airline_info_atomic
from MongoDB.Airline.upload_data import upload_country_data, upload_restricted_item_data, upload_minimum_connection_time_data
from MongoDB.Airline.upload_data import upload_airport_procedure_data, upload_transit_path_data

from MongoDB.Airport.AirportCongestionPredict import fetch_and_save_airport_congestion_predict
from MongoDB.Airport.AirportEnterprise import fetch_and_save_airport_enterprise
from MongoDB.Airport.AirportFacility import fetch_and_save_airport_facility

from MongoDB.Parking.ParkingFeeDiscountPolicy import upload_discount_policy_from_csv
from MongoDB.Parking.ParkingFeePayment import upload_payment_methods_from_csv
from MongoDB.Parking.ParkingLot import upload_parking_lot_from_csv
from MongoDB.Parking.ParkingLotWalkTime import update_parking_walk_time

from MongoDB.Weather.ATMOS import fetch_and_save_atmos_data
from MongoDB.Weather.TAF import fetch_and_save_taf_data



from dotenv import load_dotenv
from pymongo import MongoClient
import pandas as pd

# 현재 스크립트 파일의 절대 경로 기준으로 베이스 디렉토리 지정
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

EXCEL_FILES_DIR = os.path.join(BASE_DIR, "MongoDB", "Airline")
airport_list_file = os.path.join(EXCEL_FILES_DIR, "국토교통부_세계공항_정보_20241231.csv")
unmatched_csv_path = os.path.join(EXCEL_FILES_DIR, "output", "unmatched_airlines_for_schedules.csv")

PARKING_DIR = os.path.join(BASE_DIR, "MongoDB", "Parking")
parking_discount_csv_path = os.path.join(PARKING_DIR, "ParkingFeeDiscountPolicy​.csv")
parking_fee_payment_csv_path = os.path.join(PARKING_DIR, "ParkingFeePayment.csv")
parkingLot_csv_path = os.path.join(PARKING_DIR, "ParkingLot.csv")


# 로깅 설정
logging.basicConfig(
    filename="logs/scheduler.log",
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)

def run_task_with_logging(func, task_name):
    logging.info(f"작업 시작: {task_name}")
    print(f"{datetime.now(ZoneInfo("Asia/Seoul"))} - 작업 시작: {task_name}")
    try:
        func()
        logging.info(f"작업 완료: {task_name}")
        print(f"{datetime.now(ZoneInfo("Asia/Seoul"))} - 작업 완료: {task_name}")
    except Exception as e:
        logging.error(f"작업 중 오류 발생({task_name}): {e}")
        print(f"{datetime.now(ZoneInfo("Asia/Seoul"))} - 작업 중 오류 발생({task_name}): {e}")

def main():
    # 마지막 실행 시점 기록용 변수 (초 단위)
    last_run = {
        "TAF": datetime.min,
        "ATMOS": datetime.min,
        "DAILY": datetime.min,
    }

    while True:
        now = datetime.now(ZoneInfo("Asia/Seoul"))

        # 1시간마다 실행 (TAF)
        if now - last_run["TAF"] >= timedelta(hours=1):
            run_task_with_logging(fetch_and_save_taf_data, "TAF 데이터 갱신")
            last_run["TAF"] = now

        # 5분마다 실행 (ATMOS)
        if now - last_run["ATMOS"] >= timedelta(minutes=5):
            run_task_with_logging(fetch_and_save_atmos_data, "ATMOS 데이터 갱신")
            last_run["ATMOS"] = now

        # 하루 1번 실행 (나머지)
        if now - last_run["DAILY"] >= timedelta(days=1):
            
            run_task_with_logging(upload_airline_data_atomic, "airline 데이터 갱신")
            run_task_with_logging(lambda: upload_airport_data_atomic(airport_list_file), "airport 데이터 갱신")
            run_task_with_logging(fetch_flight_schedule, "flight schedule 데이터 갱신")
            run_task_with_logging(lambda: update_airline_info_atomic(unmatched_csv_path), "airline 정보 업데이트")
            run_task_with_logging(upload_country_data, "국가 데이터 갱신")
            run_task_with_logging(upload_restricted_item_data, "제한 품목 데이터 갱신")
            run_task_with_logging(upload_minimum_connection_time_data, "최소 환승 시간 데이터 갱신")
            run_task_with_logging(upload_airport_procedure_data, "공항 절차 데이터 갱신")
            run_task_with_logging(upload_transit_path_data, "환승 경로 데이터 갱신")
            
            run_task_with_logging(fetch_and_save_airport_congestion_predict, "공항 혼잡 예측 데이터 갱신")
            run_task_with_logging(fetch_and_save_airport_enterprise, "공항 기업 데이터 갱신")
            run_task_with_logging(fetch_and_save_airport_facility, "공항 시설 데이터 갱신")
            
            run_task_with_logging(lambda: upload_discount_policy_from_csv(parking_discount_csv_path), "주차 할인 정책 데이터 갱신")
            run_task_with_logging(lambda: upload_payment_methods_from_csv(parking_fee_payment_csv_path), "주차 요금 결제 방법 데이터 갱신")
            run_task_with_logging(lambda: upload_parking_lot_from_csv(parkingLot_csv_path), "주차장 데이터 갱신")
            run_task_with_logging(update_parking_walk_time, "주차장 보행 시간 데이터 갱신")




            last_run["DAILY"] = now

        # 다음 체크까지 슬립 (1분)
        time.sleep(60)

if __name__ == "__main__":
    main()
