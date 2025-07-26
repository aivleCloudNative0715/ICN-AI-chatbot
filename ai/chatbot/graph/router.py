def route_by_intent(state: dict) -> str:
    intent = state["intent"]
    # 의도명이 이미 handler 노드 이름과 동일하다고 가정하고 `_handler`를 붙임
    return f"{intent}_handler"