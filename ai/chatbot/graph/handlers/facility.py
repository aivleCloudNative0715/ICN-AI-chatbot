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

    # 🚀 최적화: slot 정보 우선 활용, 없으면 LLM fallback
    slots = state.get("slots", [])
    
    # Terminal/위치 정보 추출
    terminal_slots = [word for word, slot in slots if slot in ['B-terminal', 'I-terminal']]
    area_slots = [word for word, slot in slots if slot in ['B-area', 'I-area']]
    location_keyword = None
    
    if terminal_slots:
        location_keyword = terminal_slots[0]
        print(f"디버그: ⚡ slot에서 터미널 정보 추출: {location_keyword}")
    elif area_slots:
        location_keyword = area_slots[0]
        print(f"디버그: ⚡ slot에서 구역 정보 추출: {location_keyword}")
    else:
        print("디버그: slot에 위치 정보 없음, LLM으로 fallback")
        location_keyword = extract_location_with_llm(query_to_process)
        print(f"디버그: LLM으로 추출된 위치 정보 - {location_keyword}")
    
    # 시설명 정보 추출
    facility_slots = [word for word, slot in slots if slot in ['B-facility_name', 'I-facility_name']]
    
    if facility_slots:
        facility_names = facility_slots
        print(f"디버그: ⚡ slot에서 시설명 추출 완료 (LLM 호출 생략): {facility_names}")
    else:
        print("디버그: slot에 시설명 정보 없음, LLM으로 fallback")
        facility_names = _extract_facility_names_with_llm(query_to_process)
        print(f"디버그: LLM을 사용해 추출된 시설 목록 - {facility_names}")

    if not facility_names:
        return {**state, "response": "죄송합니다. 요청하신 시설 정보를 찾을 수 없습니다."}
    
    rag_config = RAG_SEARCH_CONFIG.get(intent_name, {})
    collection_name = rag_config.get("collection_name")
    vector_index_name = rag_config.get("vector_index_name")
    intent_description = rag_config.get("description", intent_name)
    
    # query_filter는 더 이상 사용하지 않습니다.
    # query_filter = rag_config.get("query_filter")

    if not (collection_name and vector_index_name):
        error_msg = f"죄송합니다. '{intent_name}' 의도에 대한 정보 검색 설정을 찾을 수 없거나 인덱스 이름이 누락되었습니다."
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

    individual_responses = []
    try:
        for facility_name in facility_names:
            print(f"디버그: '{facility_name}'에 대한 RAG 파이프라인 시작...")
            
            # 1단계: 넓은 벡터 검색 (필터 없이)
            query_embedding = get_query_embedding(facility_name)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                query_filter={}, # 빈 딕셔너리를 전달하여 필터링을 하지 않음
                top_k=10
            )
            print(f"디버그: '{facility_name}'에 대해 {len(retrieved_docs_text)}개 문서 검색 완료.")

            # 📌 2단계: 파이썬에서 직접 필터링
            filtered_docs = []
            if location_keyword:
                # 📌 수정된 로직: 위치 키워드에 대한 다양한 변형을 고려
                # location_variants 리스트에 키워드를 추가하여 유연하게 필터링
                location_variants = []
                if "1터미널" in location_keyword or "제1" in location_keyword or "t1" in location_keyword.lower():
                    location_variants = ["제1여객터미널", "1터미널", "T1", "1여객터미널", "1 터미널", "제1 터미널", "일터미널", "일 터미널"]
                elif "2터미널" in location_keyword or "제2" in location_keyword or "t2" in location_keyword.lower():
                    location_variants = ["제2여객터미널", "2터미널", "T2", "2여객터미널", "2 터미널", "제2 터미널", "이터미널", "이 터미널"]
                elif "탑승동" in location_keyword:
                    location_variants = ["탑승동"]
                
                # 문서가 location_variants 중 하나라도 포함하면 유효하다고 판단
                for doc in retrieved_docs_text:
                    if any(variant in doc for variant in location_variants):
                        filtered_docs.append(doc)
                print(f"디버그: '{facility_name}'에 대해 위치 필터링 후 {len(filtered_docs)}개 문서 남음.")
            else:
                filtered_docs = retrieved_docs_text

            final_context = "\n\n".join(filtered_docs)

            # 3단계: 최종 LLM 답변 생성
            if final_context:
                truncated_docs_list = final_context.split('\n\n')[:5]
                final_context_truncated = "\n\n".join(truncated_docs_list)
                
                final_response_text = common_llm_rag_caller(
                    query_to_process,
                    final_context_truncated,
                    intent_description,
                    intent_name
                )
                individual_responses.append(final_response_text)
            else:
                individual_responses.append(f"죄송합니다. 요청하신 '{location_keyword}' '{facility_name}' 정보를 찾을 수 없습니다.")

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}
    finally:
        close_mongo_client()

    final_response = _combine_individual_responses(individual_responses)
    return {**state, "response": final_response}