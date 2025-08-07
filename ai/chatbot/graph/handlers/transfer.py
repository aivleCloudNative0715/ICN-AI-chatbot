from chatbot.graph.state import ChatState

from chatbot.rag.utils import get_query_embedding, perform_vector_search, close_mongo_client
from chatbot.rag.config import RAG_SEARCH_CONFIG, common_llm_rag_caller

def transfer_info_handler(state: ChatState) -> ChatState:
    """
    'transfer_info' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 MongoDB에서 환승 관련 일반 정보를 검색하고 답변을 생성합니다.
    여러 환승 주제에 대한 복합 질문도 처리할 수 있도록 개선되었습니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "transfer_info")
    # slots 정보를 가져와서 사용합니다.
    slots = state.get("slots", [])

    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    # 슬롯에서 'B-transfer_topic' 태그가 붙은 키워드를 모두 추출합니다.
    search_keywords = [word for word, slot in slots if slot == 'B-transfer_topic']

    # 만약 슬롯에서 키워드를 찾지 못했다면, 전체 쿼리를 사용합니다.
    if not search_keywords:
        search_keywords = [user_query]
        print("디버그: 슬롯에서 환승 주제를 찾지 못했습니다. 전체 쿼리로 검색을 시도합니다.")

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
        # 추출된 각 키워드에 대해 RAG 검색을 개별적으로 수행합니다.
        for keyword in search_keywords:
            print(f"디버그: '{keyword}'에 대해 검색 시작...")

            # 키워드 임베딩 및 벡터 검색
            query_embedding = get_query_embedding(keyword)
            retrieved_docs_text = perform_vector_search(
                query_embedding,
                collection_name=collection_name,
                vector_index_name=vector_index_name,
                query_filter=query_filter,
                top_k=5 # 검색할 문서 개수
            )
            all_retrieved_docs_text.extend(retrieved_docs_text)

        print(f"디버그: MongoDB에서 총 {len(all_retrieved_docs_text)}개 문서 검색 완료.")

        # 3. 검색된 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")
        
        # 4. 공통 LLM 호출 함수를 사용하여 최종 답변 생성
        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

def transfer_route_guide_handler(state: ChatState) -> ChatState:
    """
    'transfer_route_guide' 의도에 대한 RAG 기반 핸들러.
    사용자 쿼리를 기반으로 TransitPathVector와 ConnectionTimeVector에서 환승 경로 및 최저 환승 시간 정보를 검색하고 답변을 생성합니다.
    각 컬렉션에 맞는 인덱스 이름을 사용합니다.
    """
    user_query = state.get("user_input", "")
    intent_name = state.get("intent", "transfer_route_guide")

    if not user_query:
        print("디버그: 사용자 쿼리가 비어 있습니다.")
        return {**state, "response": "죄송합니다. 질문 내용을 파악할 수 없습니다. 다시 질문해주세요."}

    print(f"\n--- {intent_name.upper()} 핸들러 실행 ---")
    print(f"디버그: 사용자 쿼리 - '{user_query}'")

    rag_config = RAG_SEARCH_CONFIG.get(intent_name, {})
    # config에서 각 컬렉션별 이름과 인덱스 정보를 명시적으로 가져옴
    main_collection_info = rag_config.get("main_collection", {})
    additional_collections_info = rag_config.get("additional_collections", [])
    intent_description = rag_config.get("description", intent_name)
    query_filter = rag_config.get("query_filter")

    # 필수 설정 확인 (메인 컬렉션 정보는 반드시 있어야 함)
    if not (main_collection_info.get("name") and main_collection_info.get("vector_index")):
        error_msg = f"죄송합니다. '{intent_name}' 의도에 대한 메인 컬렉션 설정(이름 또는 인덱스)이 누락되었습니다."
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}

    try:
        # 1. 사용자 쿼리 임베딩
        query_embedding = get_query_embedding(user_query)
        print("디버그: 쿼리 임베딩 완료.")

        all_retrieved_docs_text = []

        # 2. 메인 컬렉션에서 벡터 검색
        main_collection_name = main_collection_info["name"]
        main_vector_index = main_collection_info["vector_index"]
        print(f"디버그: 메인 컬렉션 '{main_collection_name}' (인덱스: '{main_vector_index}')에서 문서 검색 시작.")
        main_docs_text = perform_vector_search(
            query_embedding,
            collection_name=main_collection_name,
            vector_index_name=main_vector_index,
            query_filter=query_filter,
            top_k=3
        )
        all_retrieved_docs_text.extend(main_docs_text)
        print(f"디버그: 메인 컬렉션에서 {len(main_docs_text)}개 문서 검색 완료.")

        # 3. 추가 컬렉션들에서 벡터 검색
        for col_info in additional_collections_info:
            col_name = col_info.get("name")
            col_vector_index = col_info.get("vector_index")
            if col_name and col_vector_index:
                print(f"디버그: 추가 컬렉션 '{col_name}' (인덱스: '{col_vector_index}')에서 문서 검색 시작.")
                additional_docs_text = perform_vector_search(
                    query_embedding,
                    collection_name=col_name,
                    vector_index_name=col_vector_index,
                    query_filter=query_filter,
                    top_k=2
                )
                all_retrieved_docs_text.extend(additional_docs_text)
                print(f"디버그: 추가 컬렉션 '{col_name}'에서 {len(additional_docs_text)}개 문서 검색 완료.")
            else:
                print(f"디버그: 추가 컬렉션 '{col_info}' 설정이 불완전합니다. 스킵합니다.")


        if not all_retrieved_docs_text:
            print("디버그: 검색된 관련 문서가 없습니다.")
            return {**state, "response": "죄송합니다. 요청하신 환승 관련 정보를 찾을 수 없습니다."}

        # 4. 검색된 모든 문서 내용을 LLM에 전달할 컨텍스트로 결합
        context_for_llm = "\n\n".join(all_retrieved_docs_text)
        print(f"디버그: LLM에 전달될 최종 컨텍스트 길이: {len(context_for_llm)}자.")

        # 5. 공통 LLM 호출 함수를 사용하여 최종 답변 생성
        final_response = common_llm_rag_caller(user_query, context_for_llm, intent_description, intent_name)

        return {**state, "response": final_response}

    except Exception as e:
        error_msg = f"죄송합니다. 정보를 검색하는 중 오류가 발생했습니다: {e}"
        print(f"디버그: {error_msg}")
        return {**state, "response": error_msg}