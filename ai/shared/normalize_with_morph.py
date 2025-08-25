import re
from konlpy.tag import Okt

okt = Okt()

def clean_text(text):
    """
    KoBERT 기반 전처리에 적합하도록 특수문자 제거 및 공백 정리
    """
    # 한글, 영문, 숫자, 공백만 남기기
    text = re.sub(r"[^\uAC00-\uD7A3a-zA-Z0-9\s]", "", str(text))
    # 다중 공백 제거
    text = re.sub(r"\s+", " ", text)
    return text.strip()

# 플레이스홀더
FLIGHT_PREFIX = "FLIGHT"   # 토큰은 ⟪FLIGHT0⟫, ⟪FLIGHT1⟫ ... 형태로 생성
TERMINAL_PREFIX = "TERMINAL"  # 토큰은 ⟪TERMINAL0⟫, ⟪TERMINAL1⟫ ... 형태로 생성

# 항공사 코드 기반 항공편 패턴 매칭
from shared.airline_codes import AIRLINE_CODES

# 유효한 항공사 코드만 매칭하는 패턴 생성
airline_codes_pattern = '|'.join(AIRLINE_CODES)

# 일반적인 항공편 패턴 (공백 없음)
flight_pattern_normal = re.compile(rf'\b({airline_codes_pattern})\s*[-]?\s*(\d{{1,4}})\b', re.IGNORECASE)

# 띄어쓰기된 항공편 패턴 (예: "HL 7201", "7 C 0102")
flight_pattern_spaced = re.compile(r'\b([A-Za-z0-9])\s+([A-Za-z0-9])\s+(\d{1,4})\b', re.IGNORECASE)

def _collapse_flight_spans(text: str) -> str:
    """항공편 표현을 항상 붙여쓰기(하이픈/공백 제거) + 대문자로 통일."""
    # 일반 패턴 처리 (ke 907 -> KE907, KE 907 -> KE907)
    text = flight_pattern_normal.sub(lambda m: (m.group(1) + m.group(2)).upper(), text)
    
    # 띄어쓰기된 패턴 처리 (7 c 0102 -> 7C0102, hl 7201 -> HL7201)
    def spaced_replacer(m):
        code = m.group(1) + m.group(2)  # 항공사 코드 결합
        number = m.group(3)             # 항공편 번호
        # 유효한 항공사 코드인지 확인
        if code.upper() in AIRLINE_CODES:
            return (code + number).upper()  # 대문자로 변환
        return m.group(0)  # 매칭되지 않으면 원본 유지
    
    text = flight_pattern_spaced.sub(spaced_replacer, text)
    return text

def _collapse_terminal_spans(text: str) -> str:
    """터미널 표현을 T1, T2로 정규화."""
    # T1 관련 패턴들
    t1_patterns = [
        r'(?:제?\s*1\s*(?:여객\s*)?터미널|터미널\s*1|T\s*-?\s*1|첫\s*번?\s*째\s*(?:여객\s*)?터미널|제일\s*(?:여객\s*)?터미널)',
        r'(?:일\s*(?:여객\s*)?터미널|터미널\s*일)',
        r'(?:제\s*1\s*여객\s*터미널|제1\s*여객\s*터미널)',
    ]
    
    # T2 관련 패턴들  
    t2_patterns = [
        r'(?:제?\s*2\s*(?:여객\s*)?터미널|터미널\s*2|T\s*-?\s*2|두\s*번?\s*째\s*(?:여객\s*)?터미널|제이\s*(?:여객\s*)?터미널)',
        r'(?:이\s*(?:여객\s*)?터미널|터미널\s*이)',
        r'(?:제\s*2\s*여객\s*터미널|제2\s*여객\s*터미널)',
    ]
    
    # T1으로 정규화
    for pattern in t1_patterns:
        text = re.sub(pattern, 'T1', text, flags=re.IGNORECASE)
    
    # T2로 정규화  
    for pattern in t2_patterns:
        text = re.sub(pattern, 'T2', text, flags=re.IGNORECASE)
    
    return text

_FACILITY_LIST = ["기도실", "검역장", "수유실"]  # 필요하면 여기에 계속 추가
def _collapse_keyword(text: str, word: str) -> str:
    base = word.replace(" ", "")
    # 한글/영문/숫자 경계에서만 매치되게 경계 추가
    pattern = r'(?<![가-힣A-Za-z0-9])' + r'\s*'.join(map(re.escape, base)) + r'(?![가-힣A-Za-z0-9])'
    return re.sub(pattern, base, text)

def _collapse_facility_spans(text: str) -> str:
    for w in _FACILITY_LIST:
        text = _collapse_keyword(text, w)
    return text

def normalize_with_morph(text: str) -> str:
    # 0) 특수문자 제거 및 공백 정리
    processed_text = clean_text(text)
    
    # 1) 항공편을 먼저 붙여쓰기 정규화 (KE 907 -> KE907)
    processed_text = _collapse_flight_spans(processed_text)
    
    # 1.5) 터미널 표현 정규화 (1터미널 -> T1, 제2터미널 -> T2)
    processed_text = _collapse_terminal_spans(processed_text)

    # 2) 항공편을 플레이스홀더로 치환 (여러 개 지원)
    flight_map = {}  # 예: {'⟪FLIGHT0⟫': 'KE907', '⟪FLIGHT1⟫': 'VS5501'}
    flight_counter = 0
    def _flight_repl(m):
        nonlocal flight_counter
        code = (m.group(1) + m.group(2)).upper()     # 붙여쓰기 + 대문자 변환
        token = f'⟪{FLIGHT_PREFIX}{flight_counter}⟫'
        flight_map[token] = code
        flight_counter += 1
        return token

    processed_text = flight_pattern_normal.sub(_flight_repl, processed_text)
    
    # 2.5) 터미널을 플레이스홀더로 치환
    terminal_map = {}  # 예: {'⟪TERMINAL0⟫': 'T1', '⟪TERMINAL1⟫': 'T2'}
    terminal_counter = 0
    def _terminal_repl(m):
        nonlocal terminal_counter
        terminal_code = m.group(0)  # T1 또는 T2
        token = f'⟪{TERMINAL_PREFIX}{terminal_counter}⟫'
        terminal_map[token] = terminal_code
        terminal_counter += 1
        return token
    
    # T1, T2 패턴을 플레이스홀더로 치환
    terminal_pattern = re.compile(r'\bT[12]\b')
    processed_text = terminal_pattern.sub(_terminal_repl, processed_text)


    # 3) 형태소 분석 (정규화/어간화 끔)
    tokens = okt.morphs(processed_text, norm=False, stem=False)

    # 4) 다시 문자열로 합치기
    text_after = " ".join(tokens)

    # 5) 플레이스홀더 복원
    #    토크나이즈가 공백을 끼워넣어도(⟪ FLIGHT 0 ⟫) 정확히 복원되도록 처리
    for token, code in flight_map.items():
        core = token[1:-1]  # 'FLIGHT0'
        m = re.match(r'(FLIGHT)(\d+)$', core)
        if m:
            pat = re.compile(r'⟪\s*' + m.group(1) + r'\s*' + m.group(2) + r'\s*⟫')
            text_after = pat.sub(code, text_after)
        # 혹시 그대로 남아있으면 직접 치환
        text_after = text_after.replace(token, code)
    
    # 터미널 플레이스홀더 복원
    for token, code in terminal_map.items():
        core = token[1:-1]  # 'TERMINAL0'
        m = re.match(r'(TERMINAL)(\d+)$', core)
        if m:
            pat = re.compile(r'⟪\s*' + m.group(1) + r'\s*' + m.group(2) + r'\s*⟫')
            text_after = pat.sub(code, text_after)
        # 혹시 그대로 남아있으면 직접 치환
        text_after = text_after.replace(token, code)

    # 6) 혹시 남은 공백/하이픈 변형을 다시 한 번 정규화
    text_after = _collapse_flight_spans(text_after)
    text_after = _collapse_terminal_spans(text_after)

    text_after = _collapse_facility_spans(text_after)

    # 7) 공백 정리
    text_after = re.sub(r'\s+', ' ', text_after).strip()
    return text_after
