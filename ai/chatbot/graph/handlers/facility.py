from typing import List, Dict, Any
from chatbot.graph.state import ChatState
from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller
from chatbot.rag.llm_tools import extract_location_with_llm, _extract_facility_names_with_llm, _filter_and_rerank_docs

def _combine_individual_responses(responses: List[str]) -> str:
    """개별 RAG 핸들러의 응답을 하나로 합치는 헬퍼 함수"""
    if not responses:
        return "죄송합니다. 요청하신 시설 정보를 찾을 수 없습니다."

    # 응답이 하나일 경우
    if len(responses) == 1:
        return responses[0]

    # 복수일 경우 번호를 붙여 결합
    combined_text = "사용자님의 여러 질문에 대한 답변입니다.\n\n"
    for idx, response in enumerate(responses, 1):
        combined_text += f"{idx}. {response}\n"
    return combined_text

def facility_guide_handler(state: ChatState) -> ChatState:
    """
    'facility_guide' 의도에 대한 RAG 기반 핸들러.
    각 시설별로 RAG를 개별적으로 수행하고 결과를 합칩니다.
    """
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "facility_guide")

    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # 1. LLM을 사용하여 위치 정보와 시설 이름 목록 추출
    location_keyword = extract_location_with_llm(query_to_process)
    facility_names = _extract_facility_names_with_llm(query_to_process)
    print(f"디버그: LLM으로 추출된 위치 정보 - {location_keyword}")
    print(f"디버그: LLM을 사용해 추출된 시설 목록 - {facility_names}")

    if not facility_names:
        return {**state, "response": "죄송합니다. 요청하신 시설 정보를 찾을 수 없습니다."}
        
    rag_config = RAG_SEARCH_CONFIG.get(intent_name, {})
    collection_name = rag_config.get("collection_name")
    vector_index_name = rag_config.get("vector_index_name")
    intent_description = rag_config.get("description", intent_name)
    query_filter = rag_config.get("query_filter")

    if not (collection_name and vector_index_name):
        error_msg = f"죄송합니다. '{intent_name}' 의도에 대한 정보 검색 설정을 찾을 수 없거나 인덱스 이름이 누락되었습니다."
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

    individual_responses = []
    try:
        # 📌 수정된 로직: 각 시설 이름별로 완전한 RAG 파이프라인을 실행
        for facility_name in facility_names:
            print(f"디버그: '{facility_name}'에 대한 RAG 파이프라인 시작...")
            
            # 1단계: 넓은 벡터 검색
            query_embedding = get_query_embedding(facility_name)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                query_filter=query_filter,
                top_k=10
            )
            print(f"디버그: '{facility_name}'에 대해 {len(retrieved_docs_text)}개 문서 검색 완료.")

            # 2단계: 필터링 및 재정렬
            context_for_llm = "\n\n".join(retrieved_docs_text)
            final_context = context_for_llm # 필터링을 건너뛰고 모든 문서를 사용

            if not final_context and retrieved_docs_text:
                print(f"디버그: '{facility_name}' 필터링 실패. 원본 문서로 답변 생성 시도.")
                final_context = context_for_llm
            
            # 3단계: 최종 LLM 답변 생성
            if final_context:
                truncated_docs_list = final_context.split('\n\n')[:10]
                final_context_truncated = "\n\n".join(truncated_docs_list)
                
                sub_query_to_process = f"'{location_keyword}'에 있는 '{facility_name}'에 대한 정보"
                
                final_response_text = common_llm_rag_caller(
                    sub_query_to_process,
                    final_context_truncated,
                    intent_description,
                    intent_name
                )
                individual_responses.append(final_response_text)
            else:
                individual_responses.append(f"죄송합니다. 요청하신 '{location_keyword}'에 있는 '{facility_name}' 정보를 찾을 수 없습니다.")

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}
    finally:
        close_mongo_client()

    # 📌 수정된 로직: 모든 개별 응답을 하나로 합칩니다.
    final_response = _combine_individual_responses(individual_responses)
    return {**state, "response": final_response}