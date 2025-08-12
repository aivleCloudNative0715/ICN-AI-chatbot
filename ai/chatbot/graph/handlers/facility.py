# C:\Users\User\Desktop\ICN-AI-chatbot\ai\chatbot\graph\handlers\facility.py
from chatbot.graph.state import ChatState
from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller
from chatbot.rag.llm_tools import extract_location_with_llm, _extract_facility_names_with_llm, _filter_and_rerank_docs

def facility_guide_handler(state: ChatState) -> ChatState:
    """
    'facility_guide' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 공항 시설 정보를 검색하고 답변을 생성합니다.
    """
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "facility_guide")
    slots = state.get("slots", [])

    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # 1. LLM을 사용하여 위치 정보 추출
    location_keyword = extract_location_with_llm(query_to_process)
    print(f"디버그: LLM으로 추출된 위치 정보 - {location_keyword}")

    # 2. LLM을 사용하여 시설 이름 목록 추출
    facility_names = _extract_facility_names_with_llm(query_to_process)
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

    all_retrieved_docs_text = []
    try:
        # 📌 1단계: 각 시설 이름별로 벡터 검색을 넓게 수행
        for facility_name in facility_names:
            print(f"디버그: '{facility_name}'에 대해 넓은 검색 시작...")
            
            query_embedding = get_query_embedding(facility_name)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                query_filter=query_filter,
                top_k=10 # 📌 더 많은 문서를 가져오기 위해 top_k를 높게 설정
            )
            all_retrieved_docs_text.extend(retrieved_docs_text)
            
        print(f"디버그: MongoDB에서 총 {len(all_retrieved_docs_text)}개 문서 검색 완료.")

        # 📌 2단계: LLM을 사용하여 위치 정보로 필터링 및 재정렬
        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        final_context = _filter_and_rerank_docs(context_for_llm, location_keyword)

        if not final_context:
            return {**state, "response": "죄송합니다. 요청하신 시설 정보를 찾을 수 없습니다."}
            
        final_docs_list = final_context.split('\n\n')
        truncated_docs_list = final_docs_list[:5]
        
        # 다시 문자열로 합쳐서 LLM에 전달합니다.
        final_context_truncated = "\n\n".join(truncated_docs_list)
        
        print(f"디버그: 최종 답변 생성을 위해 {len(truncated_docs_list)}개 문서만 사용합니다.")

        final_response = common_llm_rag_caller(query_to_process, final_context_truncated, intent_description, intent_name)

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

    return {**state, "response": final_response}