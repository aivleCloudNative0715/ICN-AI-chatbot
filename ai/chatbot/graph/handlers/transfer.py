from chatbot.graph.state import ChatState

from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller
from chatbot.rag.transfer_route_helper import _parse_transfer_route_query_with_llm

def transfer_info_handler(state: ChatState) -> ChatState:
    """
    'transfer_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 환승 관련 일반 정보를 검색하고 답변을 생성합니다.
    여러 환승 주제에 대한 복합 질문도 처리할 수 있도록 개선되었습니다.
    """
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "transfer_info")
    slots = state.get("slots", [])

    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # 🚀 최적화: slot 정보 우선 활용, 없으면 LLM fallback
    # 슬롯에서 환승 관련 키워드를 모두 추출합니다.
    transfer_topics = [word for word, slot in slots if slot in ['B-transfer_topic', 'I-transfer_topic']]
    transport_types = [word for word, slot in slots if slot in ['B-transport_type', 'I-transport_type']]
    location_keywords = [word for word, slot in slots if slot in ['B-location', 'I-location', 'B-terminal', 'I-terminal']]
    
    search_keywords = []
    if transfer_topics or transport_types or location_keywords:
        print(f"디버그: ⚡ slot에서 환승 정보 추출 - 주제:{transfer_topics}, 교통:{transport_types}, 위치:{location_keywords}")
        
        # slot 조합으로 구체적인 검색 쿼리 생성
        all_keywords = transfer_topics + transport_types + location_keywords
        search_keywords = list(set(all_keywords)) if all_keywords else [query_to_process]
        print(f"디버그: ⚡ slot 기반 검색 키워드: {search_keywords}")
    else:
        print("디버그: slot에 환승 정보 없음, LLM으로 fallback")
        search_keywords = [query_to_process]

    # RAG_SEARCH_CONFIG에서 현재 의도에 맞는 설정 가져오기
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
        for keyword in search_keywords:
            print(f"디버그: '{keyword}'에 대해 검색 시작...")

            # 📌 수정된 부분: 검색을 위해 query_embedding에 keyword를 전달합니다.
            query_embedding = get_query_embedding(keyword)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                query_filter=query_filter,
                top_k=5
            )
            all_retrieved_docs_text.extend(retrieved_docs_text)

        print(f"디버그: MongoDB에서 총 {len(all_retrieved_docs_text)}개 문서 검색 완료.")
        
        if not all_retrieved_docs_text:
            return {**state, "response": "죄송합니다. 요청하신 환승 정보를 찾을 수 없습니다."}

        # 3. 검색된 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")
        
        # 4. 공통 LLM 호출 함수를 사용하여 최종 답변 생성
        # 📌 수정된 부분: common_llm_rag_caller에 query_to_process를 전달합니다.
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

def transfer_route_guide_handler(state: ChatState) -> ChatState:
    """
    'transfer_route_guide' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 TransitPathVector와 ConnectionTimeVector에서 환승 경로 및 최저 환승 시간 정보를 검색하고 답변을 생성합니다.
    복합 질문(여러 출발지-도착지 쌍)도 처리할 수 있도록 개선되었습니다.
    """
    # 📌 수정된 부분: rephrased_query를 먼저 확인하고, 없으면 user_input을 사용합니다.
    query_to_process = state.get("rephrased_query") or state.get("user_input", "")
    intent_name = state.get("intent", "transfer_route_guide")

    if not query_to_process:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 핸들러가 처리할 최종 쿼리 - '{query_to_process}'")

    # ⭐ LLM으로 복합 질문을 분해합니다.
    # 📌 수정된 부분: _parse_transfer_route_query_with_llm 함수에 재구성된 쿼리를 전달합니다.
    parsed_queries = _parse_transfer_route_query_with_llm(query_to_process)

    search_queries = []
    if parsed_queries and parsed_queries.get("requests"):
        search_queries = [req.get("query") for req in parsed_queries["requests"]]
    
    if not search_queries:
        # 📌 수정된 부분: 복합 질문으로 파악되지 않으면, 재구성된 쿼리를 사용합니다.
        search_queries = [query_to_process]
        print("디버그: 복합 질문으로 파악되지 않아 최종 쿼리로 검색을 시도합니다.")

    rag_config = RAG_SEARCH_CONFIG.get(intent_name, {})
    main_collection_info = rag_config.get("main_collection", {})
    additional_collections_info = rag_config.get("additional_collections", [])
    intent_description = rag_config.get("description", intent_name)
    query_filter = rag_config.get("query_filter")

    if not (main_collection_info.get("name") and main_collection_info.get("vector_index")):
        error_msg = f"죄송합니다. '{intent_name}' 의도에 대한 메인 컬렉션 설정(이름 또는 인덱스)이 누락되었습니다."
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

    all_retrieved_docs_text = []
    try:
        # ⭐ 분해된 각 질문에 대해 RAG 검색을 개별적으로 수행합니다.
        for query in search_queries:
            print(f"디버그: '{query}'에 대해 검색 시작...")
            
            # 📌 수정된 부분: 검색을 위해 query_embedding에 query를 전달합니다.
            query_embedding = get_query_embedding(query)
            print(f"디버그: '{query}' 쿼리 임베딩 완료.")

            # 메인 컬렉션에서 벡터 검색
            main_collection_name = main_collection_info["name"]
            main_vector_index = main_collection_info["vector_index"]
            main_docs_text = perform_vector_search(
                query_embedding,
                collection_name=main_collection_name,
                vector_index_name=main_vector_index,
                query_filter=query_filter,
                top_k=3
            )
            all_retrieved_docs_text.extend(main_docs_text)
            
            # 추가 컬렉션들에서 벡터 검색
            for col_info in additional_collections_info:
                col_name = col_info.get("name")
                col_vector_index = col_info.get("vector_index")
                if col_name and col_vector_index:
                    additional_docs_text = perform_vector_search(
                        query_embedding,
                        collection_name=col_name,
                        vector_index_name=col_vector_index,
                        query_filter=query_filter,
                        top_k=2
                    )
                    all_retrieved_docs_text.extend(additional_docs_text)

        if not all_retrieved_docs_text:
            print("디버그: 검색된 관련 문서가 없습니다.")
            return {**state, "response": "죄송합니다. 요청하신 환승 관련 정보를 찾을 수 없습니다."}

        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")
        # 📌 수정된 부분: common_llm_rag_caller에 query_to_process를 전달합니다.
        final_response = common_llm_rag_caller(query_to_process, context_for_llm, intent_description, intent_name)

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}