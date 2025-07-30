# tests/test_baggage_rule.py

import os
from pathlib import Path
from dotenv import load_dotenv # dotenv 모듈 import 추가

# .env 파일 로드 (현재 스크립트와 동일한 디렉토리인 tests 폴더에 .env가 있다고 가정)
dotenv_path = Path(__file__).resolve().parent / ".env" 

if not dotenv_path.exists():
    print(f"경고: .env 파일을 찾을 수 없습니다: {dotenv_path}")
    print("환경 변수가 제대로 로드되지 않을 수 있습니다.")

load_dotenv(dotenv_path=dotenv_path)

# 챗봇의 핵심 모듈 임포트
# baggage_rule_query_handler가 ai/chatbot/graph/handlers/policy.py에 있다고 가정합니다.
# 만약 다른 파일에 있다면 해당 파일로 경로를 수정해주세요.
from ai.chatbot.graph.handlers.policy import baggage_rule_query_handler
from ai.chatbot.graph.state import ChatState # ChatState를 사용하기 위해 임포트
from ai.chatbot.rag.utils import close_mongo_client # MongoDB 클라이언트 종료를 위해


def run_test_and_save_results_direct_handler(input_file_path: str, output_file_path: str):
    """
    입력 파일에서 쿼리를 읽어 'baggage_rule_query_handler'를 직접 실행하고,
    결과를 출력 파일에 저장합니다.
    의도 분류 단계를 건너뛰고 특정 핸들러의 동작을 직접 테스트할 때 사용합니다.
    """
    print(f"테스트 시작: 입력 파일 '{input_file_path}'")
    
    # 입력 파일 읽기
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            user_queries = [line.strip() for line in f_in if line.strip()]
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_file_path}'를 찾을 수 없습니다.")
        return

    # 출력 파일 준비
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write("--- 챗봇 수하물 규정 핸들러 직접 호출 테스트 결과 ---\n\n")
        
        for i, query in enumerate(user_queries):
            print(f"\n[{i+1}/{len(user_queries)}] 사용자 쿼리: '{query}'")
            f_out.write(f"--- 쿼리 {i+1}: {query} ---\n")
            
            try:
                # ChatState 객체 생성 시, intent를 "baggage_rule_query"로 강제 지정
                initial_state = ChatState(user_input=query, intent="baggage_rule_query")
                
                # classify_intent 노드나 graph.invoke()를 사용하지 않고, 핸들러를 직접 호출
                final_state = baggage_rule_query_handler(initial_state)
                
                response_text = final_state.get("response", "응답 없음")
                print(f"챗봇 응답: {response_text[:600]}...") # 긴 응답은 일부만 출력
                f_out.write(f"응답: {response_text}\n\n")
                
            except Exception as e:
                error_message = f"쿼리 '{query}' 처리 중 오류 발생: {e}"
                print(error_message)
                f_out.write(f"오류: {error_message}\n\n")
    
    print(f"\n테스트 완료: 결과가 '{output_file_path}'에 저장되었습니다.")
    close_mongo_client() # 테스트 종료 후 MongoDB 클라이언트 연결 닫기


# --- 실행 ---
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent # 현재 스크립트가 있는 디렉토리
    input_queries_file = script_dir / "baggage_queries.txt" # 수하물 쿼리 파일명
    output_results_file = script_dir / "baggage_test_results.txt" # 수하물 결과 파일명


    run_test_and_save_results_direct_handler(input_queries_file, output_results_file)