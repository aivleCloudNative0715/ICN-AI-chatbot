# tests/test_transfer_route.py

import os
from pathlib import Path
from dotenv import load_dotenv

# .env 파일 로드 (현재 스크립트와 동일한 디렉토리인 tests 폴더에 .env가 있다고 가정)
dotenv_path = Path(__file__).resolve().parent / ".env" 

if not dotenv_path.exists():
    print(f"경고: .env 파일을 찾을 수 없습니다: {dotenv_path}")
    print("환경 변수가 제대로 로드되지 않을 수 있습니다.")

load_dotenv(dotenv_path=dotenv_path)

# 챗봇의 핵심 모듈 임포트
from ai.chatbot.graph.handlers.transfer import transfer_route_guide_handler
from ai.chatbot.graph.state import ChatState
from ai.chatbot.rag.utils import close_mongo_client, get_query_embedding, perform_vector_search
from ai.chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller # RAG 설정과 LLM 호출기 필요


def run_test_and_save_results_direct_handler(input_file_path: str, output_file_path: str):
    """
    입력 파일에서 쿼리를 읽어 'transfer_route_guide_handler'를 직접 실행하고,
    검색된 개별 문서들과 최종 응답을 출력 파일에 저장합니다.
    """
    print(f"테스트 시작: 입력 파일 '{input_file_path}'")
    
    try:
        with open(input_file_path, 'r', encoding='utf-8') as f_in:
            user_queries = [line.strip() for line in f_in if line.strip()]
    except FileNotFoundError:
        print(f"오류: 입력 파일 '{input_file_path}'를 찾을 수 없습니다.")
        return

    # 출력 파일 준비
    with open(output_file_path, 'w', encoding='utf-8') as f_out:
        f_out.write("--- 챗봇 환승 경로 안내 핸들러 직접 호출 테스트 결과 ---\n\n")
        
        # 'transfer_route_guide' 의도에 대한 RAG 설정 가져오기
        intent_name = "transfer_route_guide"
        rag_config = RAG_SEARCH_CONFIG.get(intent_name, {})
        collection_name = rag_config.get("collection_name", "TransferRouteGuideVector") # 기본값 지정
        vector_index_name = rag_config.get("vector_index_name", "transferRouteGuide_vector_index") # 기본값 지정
        intent_description = rag_config.get("description", intent_name)
        
        if not (collection_name and vector_index_name):
            print(f"오류: '{intent_name}' 의도에 대한 RAG 검색 설정이 불완전합니다. 테스트를 중단합니다.")
            return


        for i, query in enumerate(user_queries):
            print(f"\n[{i+1}/{len(user_queries)}] 사용자 쿼리: '{query}'")
            f_out.write(f"--- 쿼리 {i+1}: {query} ---\n")
            
            try:
                # 1. 쿼리 임베딩 (테스트 스크립트에서 직접 수행)
                query_embedding = get_query_embedding(query)
                
                # 2. MongoDB 벡터 검색 (테스트 스크립트에서 직접 수행)
                retrieved_docs_text = perform_vector_search(
                    query_embedding,
                    collection_name=collection_name,
                    vector_index_name=vector_index_name,
                    top_k=5 # 검색할 문서 개수
                )
                print(f"디버그: MongoDB에서 {len(retrieved_docs_text)}개 문서 검색 완료.")

                f_out.write(f"--- 검색된 문서 ({len(retrieved_docs_text)}개) ---\n")
                if retrieved_docs_text:
                    for doc_idx, doc_text in enumerate(retrieved_docs_text):
                        f_out.write(f"  [문서 {doc_idx + 1}]\n")
                        f_out.write(f"  {doc_text}\n\n")
                else:
                    f_out.write("  검색된 문서 없음.\n\n")
                
                # 3. 검색된 문서 내용을 LLM에 전달할 컨텍스트로 결합 (핸들러와 동일하게)
                context_for_llm = "\n\n".join(retrieved_docs_text)

                # 4. LLM 호출 (핸들러가 이 부분을 호출할 때 사용할 동일한 함수)
                # 이 부분을 핸들러 내부에서 호출하도록 하기 위해 핸들러의 로직을 변경하지 않고
                # 여기서 직접 LLM을 호출하고, 핸들러는 단순 응답을 반환하도록 할 수도 있습니다.
                # 하지만 현재 핸들러는 RAG와 LLM 호출을 모두 포함하므로, 핸들러를 호출하는 것이 일반적입니다.

                # 여기서는 핸들러가 RAG_SEARCH_CONFIG를 사용하도록 구현되어 있음을 전제
                # 핸들러는 state.user_input만 받고 나머지는 내부적으로 처리
                initial_state = ChatState(user_input=query, intent=intent_name)
                final_state = transfer_route_guide_handler(initial_state)
                
                response_text = final_state.get("response", "응답 없음")
                print(f"챗봇 응답: {response_text[:2000]}...")
                f_out.write(f"--- 챗봇 최종 응답 ---\n")
                f_out.write(f"응답: {response_text}\n\n")
                
            except Exception as e:
                error_message = f"쿼리 '{query}' 처리 중 오류 발생: {e}"
                print(error_message)
                f_out.write(f"--- 오류 발생 ---\n")
                f_out.write(f"오류: {error_message}\n\n")
    
    print(f"\n테스트 완료: 결과가 '{output_file_path}'에 저장되었습니다.")
    close_mongo_client()


# --- 실행 ---
if __name__ == "__main__":
    script_dir = Path(__file__).resolve().parent
    input_queries_file = script_dir / "transfer_queries.txt"
    output_results_file = script_dir / "transfer_test_results.txt"

    run_test_and_save_results_direct_handler(input_queries_file, output_results_file)