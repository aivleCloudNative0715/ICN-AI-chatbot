import csv
import json
import re
from collections import defaultdict

from konlpy.tag import Okt
from shared.airline_codes import AIRLINE_CODES

# 항공편 패턴을 미리 컴파일 (성능 최적화)
airline_codes_pattern = '|'.join(AIRLINE_CODES)
FLIGHT_PATTERN = re.compile(rf'\b((?:{airline_codes_pattern})\d{{1,4}})(?:편| 편)?', re.IGNORECASE)
FLIGHT_PATTERN_NO_SPACE = re.compile(rf'((?:{airline_codes_pattern})\d{{1,4}})(?:편)?', re.IGNORECASE)
# 띄어쓰기된 항공편 패턴 추가
FLIGHT_PATTERN_SPACED = re.compile(r'\b([A-Za-z0-9])\s+([A-Za-z0-9])\s+(\d{1,4})(?:\s*편)?', re.IGNORECASE)

okt = Okt()

def remove_korean_josa(text):
    return re.sub(r'([가-힣A-Za-z]+)([은는이가에서의]*)', r'\1', text)

def extract_tokens_without_josa(text):
    tokens = okt.pos(text, norm=True, stem=False)
    result = []
    for word, tag in tokens:
        if tag != 'Josa':
            result.append(word)
    return ' '.join(result)

def extract_slots(question, intent_list):
    slots = defaultdict(list)
    
    # 형태소 분석으로 인한 띄어쓰기 문제를 해결하기 위해 공백 제거된 버전도 생성
    question_no_space = re.sub(r'\s+', '', question)

    # Rule 1: Terminal (T1, T2)
    t1_keywords = ['1터미널', 'T1', '제1터미널', '제 1 터미널', '일 터미널', '터미널 1', '터미널 일', '제일터미널', '첫번째터미널', '첫 번째 터미널', '터미널1', 'T-1', 'T 1']
    t2_keywords = ['2터미널', 'T2', '제2터미널', '제 2 터미널', '이 터미널', '터미널 2', '터미널 이', '제이터미널', '두번째터미널', '두 번째 터미널', '터미널2', 'T-2', 'T 2']
    
    # 모든 터미널 매칭을 찾기 위한 패턴들
    found_t1 = False
    found_t2 = False
    
    # T1 키워드 체크
    for keyword in t1_keywords:
        if keyword in question or keyword.replace(' ', '') in question_no_space:
            found_t1 = True
            break
    
    # T2 키워드 체크  
    for keyword in t2_keywords:
        if keyword in question or keyword.replace(' ', '') in question_no_space:
            found_t2 = True
            break
    
    # 정규식 패턴으로 추가 체크 (형태소 분석으로 분리된 경우 대비)
    if not found_t1:
        t1_pattern = re.search(r'(?:^|\s)T1(?:\s|$)', question)
        if t1_pattern:
            found_t1 = True
            
    if not found_t2:
        t2_pattern = re.search(r'(?:^|\s)T2(?:\s|$)', question)
        if t2_pattern:
            found_t2 = True
    
    # 모든 터미널을 리스트로 수집
    found_terminals = []
    if found_t1:
        found_terminals.append('T1')
    if found_t2:
        found_terminals.append('T2')
    
    if found_terminals:
        slots['terminal'] = found_terminals

    # Rule 2: Area (입국/출국 관련)
    arrival_keywords = ['입국장', '입국 게이트', '도착층', '도착 게이트']
    departure_keywords = ['출국장', '출국 게이트', '출발층', '탑승구']
    
    is_arrival = any(keyword in question or keyword.replace(' ', '') in question_no_space for keyword in arrival_keywords)
    is_departure = any(keyword in question or keyword.replace(' ', '') in question_no_space for keyword in departure_keywords)

    if is_arrival and not is_departure:
        slots['area'] = 'arrival'
    elif is_departure and not is_arrival:
        slots['area'] = 'departure'
    # If both are present, the intent might be the tie-breaker
    elif 'arrival' in intent_list:
        slots['area'] = 'arrival'
    elif 'departure' in intent_list:
        slots['area'] = 'departure'

    # Rule 2: Area (게이트 아님)
    if (re.search(r'(출국장|탑승장|보안검색대|출발)', question) or 
        re.search(r'(출국장|탑승장|보안검색대|출발)', question_no_space)):
        slots['area'] = 'departure'
    elif (re.search(r'(입국장|도착)', question) or 
          re.search(r'(입국장|도착)', question_no_space)):
        slots['area'] = 'arrival'

    # Rule 3: Gate (숫자만 허용)
    #  - "901 게이트", "901번 게이트", "게이트 901", "탑승구 901"만 매칭
    #  - 항공편 코드(예: ZE901)의 숫자를 게이트로 오인하지 않도록
    # 원본 텍스트와 공백 제거 텍스트에서 게이트 번호 검색
    pat_gate_before = (re.search(r'(?<![A-Z])\b(\d{1,3})\s*(?:번)?\s*(?:게이트|탑승구)\b', question) or
                       re.search(r'(?<![A-Z])(\d{1,3})(?:번)?(?:게이트|탑승구)', question_no_space))
    pat_gate_after = (re.search(r'(?:게이트|탑승구)\s*(?:번호|No\.?)?\s*(\d{1,3})\b', question, re.IGNORECASE) or
                      re.search(r'(?:게이트|탑승구)(?:번호|No\.?)?(\d{1,3})', question_no_space, re.IGNORECASE))

    gate = None
    if pat_gate_before:
        gate = pat_gate_before.group(1)
    elif pat_gate_after:
        gate = pat_gate_after.group(1)

    if gate is not None:
        slots['gate'] = gate

    # Rule 4: Airline Flight - 항공사 코드 기반 매칭 (최적화)
    # 미리 컴파일된 패턴 사용
    flight_matches = FLIGHT_PATTERN.findall(question)
    # 대소문자 통일을 위해 모든 매칭을 대문자로 변환
    flight_matches = [match.upper() for match in flight_matches if match.upper()[:2] in AIRLINE_CODES]
    
    # 띄어쓰기된 패턴 매칭 (HL 7201, 7 C 0102 같은 경우)
    if not flight_matches:
        spaced_matches = FLIGHT_PATTERN_SPACED.findall(question)
        for match in spaced_matches:
            code = match[0] + match[1]  # 항공사 코드 결합
            number = match[2]           # 항공편 번호
            if code.upper() in AIRLINE_CODES:
                flight_matches.append((code + number).upper())
    
    # 공백 제거 텍스트에서 매칭 (마지막 시도)
    if not flight_matches:
        raw_matches = FLIGHT_PATTERN_NO_SPACE.findall(question_no_space)
        flight_matches = [match.upper() for match in raw_matches if match.upper()[:2] in AIRLINE_CODES]
    
    # 주차장 관련 의도에서는 주차장 구역 코드를 항공편으로 오인하지 않도록 필터링
    if flight_matches and intent_list in ['parking_walk_time_info', 'parking_availability_query', 'parking_congestion_prediction']:
        # 주차장 컨텍스트에서 나타나는 키워드들이 있으면 항공편으로 보지 않음
        parking_context_keywords = ['주차', '구역', '지하', '지상', '층', '위치', '도보', '걷', '체크인 카운터']
        has_parking_context = any(keyword in question or keyword in question_no_space for keyword in parking_context_keywords)
        
        if has_parking_context:
            # 주차장 구역 코드 패턴 (A~Z + 숫자)
            filtered_matches = []
            for match in flight_matches:
                # 실제 항공편 코드 같은 패턴만 유지 (예: KE123은 유지, H23은 제외)
                if not re.match(r'^[A-Z]\d+$', match) or len(match) > 4:
                    filtered_matches.append(match)
            flight_matches = filtered_matches
    
    if flight_matches:
        slots['flight_id'] = flight_matches  # 이미 대문자로 변환됨

    # Rule 5: Airport Code (Origin/Destination)
    # This rule should be more sophisticated to differentiate origin and destination
    # For now, it captures any 3-letter airport code.
    airport_code_matches = re.findall(r'\b([A-Z0-9]{2, 3})\b', question)
    if airport_code_matches:
        # If there are multiple, we might need more context to differentiate
        # For simplicity, let\'s just take the first one for now or all of them as a list
        slots['airport_code'] = list(set(airport_code_matches)) # Use set to avoid duplicates

    # NEW Rule for Flight Status
    flight_status_keywords = ['지연', '결항', '취소', '연착', '정시']
    flight_related_keywords = ['항공편', '비행기', '항공기', '항공권', '운항']
    if (any(keyword in question or keyword.replace(' ', '') in question_no_space for keyword in flight_related_keywords) or 
        'flight_id' in slots):
        if '도착' in question or '도착' in question_no_space:
            slots['flight_status'] = '도착'
        elif '출발' in question or '출발' in question_no_space:
            slots['flight_status'] = '출발'

    for status in flight_status_keywords:
        if status in question or status in question_no_space:
            slots['flight_status'] = status
            break

    airports = ['인천', '김포', '제주', '상하이', '도쿄', '뉴욕', '파리', '런던', '프랑크푸르트', '방콕', 'LA', '시드니', '헬싱키',
                '후쿠오카', '쿠알라룸푸르', '마닐라', '모스크바', '아부다비', '밴쿠버', '달라스', '삿포로', '런던 히드로', '싱가포르']

    for ap in airports:
        if f'{ap}발' in question or f'{ap}발' in question_no_space:
            slots['departure_airport'] = ap
        if f'{ap}행' in question or f'{ap}행' in question_no_space:
            slots['arrival_airport'] = ap

    # Rule 6: Airline Name (New Rule)
    airline_keywords = [
        '대한항공', '아시아나항공', '제주항공', '진에어', '티웨이항공', '에어부산', '이스타항공',
        '에어서울', '델타항공', '유나이티드항공', '아메리칸항공', '루프트한자', '에어프랑스',
        'KLM', '싱가포르항공', '캐세이퍼시픽', '콴타스항공', '에미레이트항공', '카타르항공',
        '터키항공', '핀에어', 'LOT폴란드항공', '에티하드항공', '에어캐나다', '중국동방항공',
        '중국남방항공', '중국국제항공', '일본항공', '전일본공수', '타이항공', '베트남항공',
        '필리핀항공', '말레이시아항공', '가루다인도네시아', '에어아시아', '라이언에어',
        '이지젯', '스카이팀', '스타얼라이언스', '원월드', 'FedEX항공', 'FedEX', '가루다항공', 
        '아시아나', '아시아나 항공', '티웨이'
    ]
    found_airline = None
    # Sort by length to match longer names first (e.g., "대한항공" before "대한")
    for airline in sorted(airline_keywords, key=len, reverse=True):
        if airline in question or airline.replace(' ', '') in question_no_space:
            slots['airline_name'] = airline
            found_airline = airline
            break


    # Rule 6: Facility Name (Enhanced with keywords)
    base_facilities = [
        '약국', '은행', '환전소', '수유실', '유아휴게실', '기도실', '면세점', '흡연실', '식당', '카페', '편의점',
        '로밍센터', '셔틀버스', '택배', '병원', '의료', '안내 데스크', '여권민원실', '병무청', '검역장', 'ATM기',
        '밥집', '의료관광 안내센터', '패스트푸드점', '한식', '정형외과', '내과', '피부과', '치과', '비즈니스센터',
        '수하물보관소', '우체국', '라운지', '여객터미널', '렌터카', '모노레일', '전망대', '샤워실', '찜질방',
        '이발소', '미용실', '호텔', '에스컬레이터', '엘리베이터', '에어트레인', '스카이워크', '자동체크인', '출입국관리소',
        '의료센터', '헬스케어센터', 'VR체험존', '키즈존', '문화센터', '전시관', '공항철도', '지하철역', 'KTX역',
        '공항버스정류장', '택시승강장', '주차장', '주차타워', '공항 안내소', '어린이 놀이시설','해외감염병신고센터',
        '환승장', '환승호텔', 'AED', 'ATM', '편의시설', '화장실'
    ]

    # 주차장 관련은 parking_lot 슬롯으로 처리되므로 facility_name에서 제외
    parking_related_facilities = ['주차장', '주차타워']
    
    for facility in base_facilities:
        if facility in question or facility.replace(' ', '') in question_no_space:
            # 주차장 관련 시설은 이미 parking_lot으로 처리되므로 제외
            if facility not in parking_related_facilities:
                slots['facility_name'].append(facility)

    if '국내선' in question or '국제선' in question or '국내선' in question_no_space or '국제선' in question_no_space:
        if '국내선' in question or '국내선' in question_no_space:
            if re.search(r'(국내선.*(에서|출발|타고|발))', question) or re.search(r'(국내선.*(에서|출발|타고|발))', question_no_space):
                slots['departure_type'] = '국내선'
            if re.search(r'(국내선.*(으로|가는|도착|가야|향해))', question) or re.search(r'(국내선.*(으로|가는|도착|가야|향해))', question_no_space):
                slots['arrival_type'] = '국내선'

        if '국제선' in question or '국제선' in question_no_space:
            if re.search(r'(국제선.*(에서|출발|타고|발))', question) or re.search(r'(국제선.*(에서|출발|타고|발))', question_no_space):
                slots['departure_type'] = '국제선'
            if re.search(r'(국제선.*(으로|가는|도착|가야|향해))', question) or re.search(r'(국제선.*(으로|가는|도착|가야|향해))', question_no_space):
                slots['arrival_type'] = '국제선'

    # 시설
    facility_keywords = ['이동통로', '탑승동', '수속']
    for keyword in facility_keywords:
        if keyword in question or keyword in question_no_space:
            slots['location_keyword'] = keyword
            break

    # Rule 7: Date/Time (Enhanced)
    date_match_md = re.search(r'(\d{1,2}월\s?\d{1,2}일)', question)
    if date_match_md:
        slots['date'] = date_match_md.group(1).strip()
    elif '오늘' in question or '오늘' in question_no_space:
        slots['date'] = '오늘'
    elif '내일' in question or '내일' in question_no_space:
        slots['date'] = '내일'
    elif '모레' in question or '모레' in question_no_space:
        slots['date'] = '모레'
    elif '어제' in question or '어제' in question_no_space:
        slots['date'] = '어제'

    day_of_week_keywords = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    for day in day_of_week_keywords:
        if day in question or day in question_no_space:
            slots['day_of_week'] = day
            break

    time_match_h_hm = re.search(r'((?P<ampm>오전|오후|아침|저녁|밤|새벽)?\s*(?P<hour>\d{1,2})시(?:\s*(?P<minute>\d{1,2}|반)분?)?)',
                                question)
    relative_time_match = re.search(
        r'(?P<number>\d{1,2}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)\s*(?P<unit>시간|분|일|주|개월)\s*(?P<indicator>뒤|후|전|동안|내에|만에)?',
        question)
    vague_time_keywords = ['곧', '이따가', '지금', '현재', '조금 뒤']
    time_period_keywords = ['오전', '오후', '아침', '저녁', '밤', '새벽']

    # 로직의 우선순위를 조정하여 relative_time을 먼저 확인
    if relative_time_match:
        number_str = relative_time_match.group('number')
        unit = relative_time_match.group('unit')
        indicator = relative_time_match.group('indicator')

        # 한글 숫자를 숫자로 변환하는 로직
        number_map = {'한': 1, '두': 2, '세': 3, '네': 4, '다섯': 5, '여섯': 6, '일곱': 7, '여덟': 8, '아홉': 9, '열': 10}
        number = int(number_str) if number_str and number_str.isdigit() else number_map.get(number_str, None)

        if number:
            relative_time_text = f"{number}{unit}"
            if indicator:
                relative_time_text += f" {indicator}"
            slots['relative_time'] = relative_time_text

    elif time_match_h_hm:
        # 시간(hour/minute) 정보 추출 로직
        hour = time_match_h_hm.group('hour')
        minute = time_match_h_hm.group('minute')
        ampm = time_match_h_hm.group('ampm')

        if ampm:
            slots['time_period'] = ampm.strip()

        if hour:
            slots['hour'] = hour.strip()

        if minute:
            if '반' in minute:
                slots['minute'] = '30'
            else:
                slots['minute'] = minute.strip()

    # 위 두 정규식에 모두 해당하지 않을 경우, 키워드 기반으로 추출
    else:
        extracted_vague_time = False
        for vague_time in vague_time_keywords:
            if vague_time in question or vague_time in question_no_space:
                slots['vague_time'] = vague_time
                extracted_vague_time = True
                break

        if not extracted_vague_time:
            for period in time_period_keywords:
                if period in question or period in question_no_space:
                    if '시' not in question and '시' not in question_no_space:
                        slots['time_period'] = period
                        break

    season_keywords = {
        '봄': ['봄', '봄철', '춘계', '꽃피는'],
        '여름': ['여름', '여름철', '하계', '무더운'],
        '가을': ['가을', '가을철', '추계', '단풍'],
        '겨울': ['겨울', '겨울철', '동계', '눈 오는', '추운']
    }

    for season, keywords in season_keywords.items():
        for word in keywords:
            if word in question or word in question_no_space:
                slots['season'] = word
                break

    if "airport_info" in intent_list:
        cleaned = extract_tokens_without_josa(question)

        codes = re.findall(r'\b([A-Z]{3})\b', cleaned)
        if codes:
            slots['airport_code'] = list(set(code.upper() for code in codes))

        name_match = re.search(r'([가-힣A-Za-z\s]+?(?:국제)?공항)', cleaned)
        if name_match:
            name = name_match.group(1).strip()

            if not (
                    re.match(r'^[A-Z]{3}', name) or
                    re.search(r'(무슨|어디|어떤|나라|있는)', name)
            ):
                slots['airport_name'] = name

    if 'airport_weather_current' in intent_list:
        weather_topics = [
            '기온', '온도', '습도', '풍속', '바람', '날씨', '강수량', '비', '눈', '시정', '안개', '미세먼지',
            '황사', '뇌우', '태풍', '운고', '기압', '이슬점', '우박', '우산', '시야', '폭우', '구름', 'TAF'
        ]
        found_weather_topics = []
        for topic in sorted(weather_topics, key=len, reverse=True):
            if topic in question:
                if not any(topic in f for f in found_weather_topics):
                    found_weather_topics.append(topic)

        if found_weather_topics:
            if found_weather_topics == ['날씨']:
                slots['weather_topic'] = ['전체']
            else:
                # 날씨는 전체 요청일 수 있으므로, 다른 주제가 있으면 날씨는 제거
                if '날씨' in found_weather_topics:
                    found_weather_topics = [
                        topic for topic in found_weather_topics if topic != '날씨'
                    ]
                # 다른 세부 주제가 있다면 그걸 유지
                slots['weather_topic'] = found_weather_topics
        else:
            # 아무 기상 주제도 없지만, 전체 요청처럼 보이는 경우
            if re.search(r'(기상\s*(정보|예보|전문|데이터|업데이트|상황))', question):
                slots['weather_topic'] = ['전체']

    # NEW Rule for Parking Slots (General) - 원본과 공백 제거 버전 모두 체크
    # 우선순위: 구체적 주차장 > 일반 주차장 (중복 방지)
    parking_patterns = [
        r'(P[1-5])',  # 가장 구체적
        r'(장기주차장|단기주차장|화물터미널 주차장|주차타워)',  # 중간 구체성
        r'(주차장)'  # 가장 일반적
    ]
    
    parking_lot_found = None
    for pattern in parking_patterns:
        match = (re.search(pattern, question) or 
                re.search(pattern.replace(' ', ''), question_no_space))
        if match:
            parking_lot_found = match.group(1)
            break
    
    if parking_lot_found:
        slots['parking_lot'] = parking_lot_found

    parking_type_keywords = {
        '장기': ['장기주차장', '장기 주차'],
        '단기': ['단기주차장', '단기 주차'],
        '화물': ['화물터미널 주차장', '화물 주차'],
        '예약': ['예약 주차장', '예약 주차']
    }
    for p_type, keywords in parking_type_keywords.items():
        if any(keyword in question or keyword in question_no_space for keyword in keywords):
            slots['parking_type'] = p_type
            break


    # NEW Rule for Parking Fee Info
    if 'parking_fee_info' in intent_list:
        duration_match = re.search(r'(\d+)\s*(시간|분|일)', question)
        if duration_match:
            slots['parking_duration_value'] = duration_match.group(1)
            slots['parking_duration_unit'] = duration_match.group(2)
        
        payment_method_keywords = ['카드', '현금', '하이패스', '모바일', '간편결제']
        for method in payment_method_keywords:
            if method in question or method in question_no_space:
                slots['payment_method'] = method
                break


    # Unified Rule for Parking Availability and Congestion
    if any(i in intent_list for i in ['parking_availability_query', 'parking_congestion_prediction']):
        # Parking Type
        if '단기 주차장' in question or '단기주차장' in question or '단기주차장' in question_no_space:
            slots['parking_type'] = '단기'
        elif '장기 주차장' in question or '장기주차장' in question or '장기주차장' in question_no_space:
            slots['parking_type'] = '장기'

        # Parking Area (specific sections)
        parking_area_keywords = [
            'P1', 'P2', 'P3', 'P4', 'P5', '지하 1층', '지하 2층', '지상 1층', '지상 2층',
            '지하', '지상', '동편', '서편', '화물터미널 주차장'
        ]
        for area in sorted(parking_area_keywords, key=len, reverse=True):
            if area in question or area in question_no_space:
                slots['parking_area'] = area
                break
        
        # Availability Status (for availability query)
        if 'parking_availability_query' in intent_list:
            availability_keywords = ['만차', '혼잡', '여유', '가능', '비어있', '꽉 찼', '주차 가능', '만차 여부', '자리 있'
                                     '자리 없', '자리 확인']
            for status in availability_keywords:
                if status in question or status in question_no_space:
                    slots['availability_status'] = status
                    break

        # Date/Time (re-using existing rules for consistency)
        date_match_md = re.search(r'(\d{1,2}월\s?\d{1,2}일)', question)
        if date_match_md:
            slots['date'] = date_match_md.group(1).strip()
        elif '오늘' in question or '오늘' in question_no_space:
            slots['date'] = '오늘'
        elif '내일' in question or '내일' in question_no_space:
            slots['date'] = '내일'
        elif '모레' in question or '모래' in question_no_space:
            slots['date'] = '모레'
        elif '어제' in question or '어제' in question_no_space:
            slots['date'] = '어제'

        day_of_week_keywords = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        for day in day_of_week_keywords:
            if day in question or day in question_no_space:
                slots['day_of_week'] = day
                break

        time_match_hm = re.search(r'((오전|오후|아침|저녁|밤|새벽)?\s?\d{1,2}시\s?\d{1,2}분)', question)
        time_match_h = re.search(r'((오전|오후|아침|저녁|밤|새벽)?\s?\d{1,2}시(\s?반)?)', question)
        relative_time_match = re.search(r'(\d{1,2}|한|두|세|네|다섯|여섯|일곱|여덟|아홉|열)?\s*(시간|분)\s*(뒤|후|전)', question)
        vague_time_keywords = ['곧', '이따가', '지금', '현재', '조금 뒤']
        time_period_keywords = ['오전', '오후', '아침', '저녁', '밤', '새벽']

        if time_match_hm:
            slots['time'] = time_match_hm.group(1).strip()
        elif time_match_h:
            slots['time'] = time_match_h.group(1).strip()
        elif relative_time_match:
            slots['time'] = ''.join([g if g else '' for g in relative_time_match.groups()]).strip()
        else:
            for vague_time in vague_time_keywords:
                if vague_time in question or vague_time in question_no_space:
                    slots['time'] = vague_time
                    break
            else:
                for period in time_period_keywords:
                    if (period in question or period in question_no_space) and '시' not in question and '시' not in question_no_space:
                        slots['time'] = period
                        break



    # 확장된 도시/국가 목록
    destinations = [
        # 일본
        '일본', '도쿄', '오사카', '나고야', '후쿠오카', '삿포로', '오키나와', '교토', '고베',
        '히로시마', '센다이', '가나자와', '니가타', '마츠야마', '요코하마', '기타큐슈', '간사이', '나리타', '하네다',


        # 중국/대만
        '중국', '베이징', '상하이', '광저우', '선전', '청두', '시안', '하얼빈', '대련',
        '항저우', '난징', '쿤밍', '우한', '톈진', '칭다오', '대만', '타이베이', '가오슝',

        # 동남아시아
        '싱가포르', '방콕', '태국', '베트남', '호치민', '하노이', '다낭', '필리핀',
        '마닐라', '세부', '말레이시아', '쿠알라룸푸르', '인도네시아', '자카르타', '발리', '호찌민', '푸동',

        # 기타 아시아
        '홍콩', '마카오', '몽골', '울란바토르', '인도', '뉴델리', '뭄바이',

        # 중동
        '두바이', 'UAE', '아부다비', '카타르', '도하', '사우디아라비아', '리야드',
        '쿠웨이트', '바레인', '오만', '터키', '이스탄불',

        # 유럽
        '독일', '프랑크푸르트', '베를린', '뮌헨', '프랑스', '파리', '영국', '런던',
        '이탈리아', '로마', '밀라노', '스페인', '마드리드', '바르셀로나', '네덜란드',
        '암스테르담', '벨기에', '브뤼셀', '스위스', '취리히', '오스트리아', '비엔나',
        '체코', '프라하', '헝가리', '부다페스트', '폴란드', '바르샤바', '러시아',
        '모스크바', '핀란드', '헬싱키', '노르웨이', '오슬로', '스웨덴', '스톡홀름',

        # 북미
        '미국', '뉴욕', 'LA', '로스앤젤레스', '시애틀', '시카고', '라스베이거스',
        '샌프란시스코', '워싱턴', '보스턴', '애틀랜타', '달라스', '휴스턴', '마이애미',
        '캐나다', '토론토', '밴쿠버', '몬트리올',

        # 오세아니아
        '호주', '시드니', '멜버른', '브리즈번', '퍼스', '뉴질랜드', '오클랜드',

        # 기타
        '브라질', '상파울루', '남아프리카공화국', '케이프타운', '이집트', '카이로',

        # 국내
        '제주', '부산', '대구', '광주', '울산',
        '김포', '김해','양양', '무안', '사천', '여수', '원주'
    ]

    found_airport_name = None  # 핸들러의 airport_name (목적지)
    found_departure_airport_name = None  # 핸들러의 departure_airport_name (출발지)

    # ========== 1단계: 출발지 패턴 우선 확인 ==========
    departure_patterns = [
        r'([가-힣A-Za-z]+)\s*에서\s*출발',  # "호찌민에서 출발"
        r'([가-힣A-Za-z]+)\s*(공항)*에서\s*오는',  # "호찌민에서 오는"
        r'([가-힣A-Za-z]+)\s*출발\s*항공편',  # "호찌민 출발 항공편"
        r'([가-힣A-Za-z]+)\s*출발\s*비행기',  # "호찌민 출발 비행기"
        r'([가-힣A-Za-z]+)발\s*항공편',  # "호찌민발 항공편"
        r'([가-힣A-Za-z]+)발\s*비행기',  # "호찌민발 비행기"
        # 형태소 분석으로 인한 띄어쓰기 버전들
        r'([가-힣A-Za-z]+)에서\s*출\s*발',  
        r'([가-힣A-Za-z]+)\s*출\s*발\s*항공\s*편',
        r'([가-힣A-Za-z]+)\s*출\s*발\s*비행\s*기',
        r'([가-힣A-Za-z]+)발\s*항공\s*편',
        r'([가-힣A-Za-z]+)발\s*비행\s*기',
    ]

    for pattern in departure_patterns:
        match = re.search(pattern, question)
        if not match:
            # 공백 제거된 텍스트에서도 시도
            match = re.search(pattern, question_no_space)
        if match:
            candidate = match.group(1).strip()
            if candidate in destinations:
                # 이미 항공사 이름으로 추출된 경우, 해당 도시는 건너뛰기
                if found_airline and candidate in found_airline:
                    continue
                found_departure_airport_name = candidate
                break

    # ========== 2단계: 목적지 패턴 확인 (출발지가 없을 때만) ==========
    if not found_departure_airport_name:
        destination_patterns = [
            r'([가-힣A-Za-z]+)\s+가는',  # "홍콩 가는"
            r'([가-힣A-Za-z]+)가는',  # "홍콩가는"
            r'([가-힣A-Za-z]+)행(?=[가-힣A-Za-z])',  # "대만행비행기" - "행" 뒤에 다른 글자가 있는 경우
            r'([가-힣A-Za-z]+)\s*행\s*$',  # "태국행" - "행"으로 끝나는 경우
            r'([가-힣A-Za-z]+)\s*행\s+',  # "태국 행 " - "행" 뒤에 공백이 있는 경우
            r'([가-힣A-Za-z]+)으로\s*가는',  # "홍콩으로 가는"
            r'([가-힣A-Za-z]+)로\s*가는',  # "홍콩로 가는"
            r'([가-힣A-Za-z]+)에\s*가는',  # "홍콩에 가는"
            r'([가-힣A-Za-z]+)\s*가고\s*싶',  # "홍콩가고싶"
            r'([가-힣A-Za-z]+)\s*여행',  # "홍콩여행"
            # 형태소 분석으로 인한 띄어쓰기 버전들
            r'([가-힣A-Za-z]+)\s+가\s*는',  
            r'([가-힣A-Za-z]+)가\s*는',
            r'([가-힣A-Za-z]+)으로\s*가\s*는',
            r'([가-힣A-Za-z]+)로\s*가\s*는',
            r'([가-힣A-Za-z]+)에\s*가\s*는',
            r'([가-힣A-Za-z]+)\s*가고\s*싶',
            r'([가-힣A-Za-z]+)\s*여\s*행',
        ]

        for pattern in destination_patterns:
            match = re.search(pattern, question)
            if not match:
                # 공백 제거된 텍스트에서도 시도
                match = re.search(pattern, question_no_space)
            if match:
                candidate = match.group(1).strip()
                if candidate in destinations:
                    # 이미 항공사 이름으로 추출된 경우, 해당 도시는 건너뛰기
                    if found_airline and candidate in found_airline:
                        continue
                    found_airport_name = candidate
                    break

    # ========== 3단계: 패턴 실패 시 키워드 검색 ==========
    if not found_departure_airport_name and not found_airport_name:
        # 출발지 키워드 우선 확인
        departure_keywords = ['에서 출발', '에서 오는', '발 항공편', '발 비행기', '출발']
        has_departure_keyword = any(keyword in question for keyword in departure_keywords)

        # 목적지 키워드 확인
        destination_keywords = ['가는', '행', '가고 싶', '여행']
        has_destination_keyword = any(keyword in question for keyword in destination_keywords)

        # 키워드에 따라 우선순위 결정
        for dest in sorted(destinations, key=len, reverse=True):
            if dest in question or dest in question_no_space:
                # 이미 항공사 이름으로 추출된 경우, 해당 도시는 건너뛰기
                if found_airline and dest in found_airline:
                    continue
                    
                if has_departure_keyword and not has_destination_keyword:
                    found_departure_airport_name = dest
                elif has_destination_keyword and not has_departure_keyword:
                    found_airport_name = dest
                elif not has_departure_keyword and not has_destination_keyword:
                    # 애매한 경우 문맥으로 판단 (기본은 목적지)
                    found_airport_name = dest
                break

    # ========== 슬롯 설정 (핸들러 변수명에 맞춰서) ==========
    if found_airport_name:
        slots['airport_name'] = found_airport_name  # 핸들러에서 사용하는 변수명

    if found_departure_airport_name:
        slots['departure_airport_name'] = found_departure_airport_name

    # NEW Rule for Baggage Rule
    if 'baggage_rule_query' in intent_list:
        baggage_types = {
            '기내': ['기내', '들고 타는'],
            '위탁': ['위탁', '부치는', '보내는', '화물칸'],
            '특수': ['스포츠 장비', '골프채', '스키', '악기', '유모차', '휠체어']
        }
        rule_types = {
            '크기': ['크기', '사이즈'],
            '무게': ['무게', '킬로그램', 'kg'],
            '개수': ['개수', '몇 개'],
            '요금': ['요금', '비용', '가격'],
            '금지 품목': ['금지', '안 되는', '반입 금지']
        }
        items = ['액체', '전자담배', '배터리', '라이터', '노트북',
                 '카메라', '음식', '약', '화장품', '전자제품', '인화성 물질',
                 '과일', '육류', '반려동물 동반', '생선', '해산물', '산소캔', 
                 '산소', '세면도구', '스노우보드', '베개', '가위'
                ]

        luggage_terms = ['수하물', '짐', '가방', '캐리어', '백팩']
        found_luggage_terms = [term for term in luggage_terms if term in question or term in question_no_space]
        if found_luggage_terms:
            slots['luggage_term'] = found_luggage_terms

        for b_type, keywords in baggage_types.items():
            if any(keyword in question or keyword in question_no_space for keyword in keywords):
                slots['baggage_type'] = b_type
                break

        for r_type, keywords in rule_types.items():
            if any(keyword in question or keyword in question_no_space for keyword in keywords):
                slots['rule_type'] = r_type
                break

        found_items = [item for item in items if item in question or item in question_no_space]
        if found_items:
            slots['item'] = found_items

    # NEW Rule: Self Bag Drop 여부 확인
    if 'baggage_rule_query' in intent_list:
        self_bag_drop_keywords = [
            '셀프 백드랍', '셀프 백드롭', 'self bag drop', 'self baggage drop', '셀프 수하물 위탁'
        ]
        for keyword in self_bag_drop_keywords:
            if keyword in question.lower() or keyword in question_no_space.lower():
                slots['self_bag_drop'] = True
                break

    # NEW Rule for Baggage Claim Info
    if 'baggage_claim_info' in intent_list:
        # Baggage belt number
        belt_match = re.search(r'(\d+)\s?[번]?\s?(수취대|컨베이어 벨트)', question)
        if belt_match:
            slots['baggage_belt_number'] = belt_match.group(1).strip()

        # Baggage issue (lost, damaged, delayed)
        if any(keyword in question or keyword in question_no_space for keyword in ['안 나오는', '분실', '못 찾']):
            slots['baggage_issue'] = '분실'
        elif '파손' in question or '파손' in question_no_space:
            slots['baggage_issue'] = '파손'
        elif '늦어' in question or '늦어' in question_no_space:
            slots['baggage_issue'] = '지연'
        
        # Baggage type (general, special, excess)
        if any(keyword in question or keyword in question_no_space for keyword in ['일반 수하물', '일반 짐']):
            slots['baggage_type'] = 'general'
        elif any(keyword in question or keyword in question_no_space for keyword in ['유모차', '휠체어', '특수 수하물', '스포츠 장비', '악기']):
            slots['baggage_type'] = 'special'
        elif any(keyword in question or keyword in question_no_space for keyword in ['초과 수하물', '추가 짐', '무게 초과']):
            slots['baggage_type'] = 'excess'

        luggage_keywords = ['짐', '수하물', '캐리어', '가방', '화물']

        for term in luggage_keywords:
            if term in question or term in question_no_space:
                slots['luggage_term'] = [term]
                break

    # NEW Rule: Transfer-related topics
    if 'transfer_info' in intent_list:
        transfer_keywords = {
            'stopover': ['스탑오버', 'stopover'],
            'layover_program': ['관광 프로그램', '투어 프로그램', '트랜짓 투어', '경유 관광'],
            'shuttle': ['셔틀', '공항 셔틀', '셔틀버스'],
            'LAGs': ['LAGs', '액체류', '젤류', '에어로졸', '액체 젤', '보안검색 액체'],
            'health_declaration': ['건강상태질문서', '건강 상태 질문서', '건강 상태 신고서'],
            'customs': ['전자세관신고', '세관 신고', '세관 심사', '관세 신고', '면세품 신고'],
            'airlines': ['항공사', '항공사 목록', '제휴 항공사']
        }

        for key, keywords in transfer_keywords.items():
            for keyword in keywords:
                if keyword in question or keyword in question_no_space:
                    slots['transfer_topic'] = key
                    break
            if 'transfer_topic' in slots:
                break

    if 'immigration_policy' in intent_list:
        policy_keywords = {
            'passport_type': ['관용여권', '일반여권', '전자여권', '긴급여권'],
            'person_type': ['국외체류자', '외국인', '내국인', '병역대상자'],
            'item': ['귀금속', '화폐', '미술품', '문화재'],
            'facility_name': ['여권민원센터', '감정관', '민원실'],
            'organization': ['외교부', '병무청', '출입국관리소'],
        }

        for slot_name, keywords in policy_keywords.items():
            found = []
            for keyword in keywords:
                if keyword in question or keyword in question_no_space:
                    found.append(keyword)

            if found:
                slots[slot_name] = found if len(found) > 1 else found[0]

    if 'parking_walk_time_info' in intent_list:
        destination_match = re.search(r'(1터미널|2터미널|T1|T2|출국장|입국장)', question)
        if destination_match:
            slots['destination'] = destination_match.group(1)

        # 출발지: 주차 위치
        origin_match = re.search(r'(T[12])?\s*(지상|지하)?\s?(\d층)?\s?(동편|서편)?\s?(\d{2,3}지점|[A-Z]\s?구역)?', question)
        if origin_match:
            origin_parts = [part for part in origin_match.groups() if part]
            if origin_parts:
                slots['origin'] = ' '.join(origin_parts).replace(' ', '')  # 공백 제거

        # 도착지: 체크인 카운터
        counter_match = re.search(r'체크인\s*카운터\s*([A-Z])', question)
        if not counter_match:
            counter_match = re.search(r'([A-Z])\s*카운터', question)
        if counter_match:
            slots['destination'] = f"카운터 {counter_match.group(1)}"

    # Rule 8: Keyword-to-Slot Mapping
    keyword_to_slot_map = {
        'airline_info_query': {
            '항공사 고객센터': 'airline_info',
            '항공사 전화번호': 'airline_info',
            '항공사 연락처': 'airline_info',
            '로고': 'airline_info',
            '이미지': 'airline_info',
            '항공사 코드': 'airline_info',
            '항공사 취항': 'airline_info',
            '취항': 'airline_info',
            '항공사': 'airline_info',
            '항공사 목록': 'airline_info',
            '화물 운송': 'airline_info',
            '항공': 'airline_info'
        },
        'parking_fee_info': {
            '주차 요금': 'fee_topic',
            '주차비': 'fee_topic',
            '주차 요금 할인': 'fee_topic',
            '주차 요금 결제': 'fee_topic',
            '요금': 'fee_topic',
            '주차 할인': 'fee_topic',
            '주차 감면': 'fee_topic',
            '주차 비용': 'fee_topic',
            '주차료': 'fee_topic',
        },
        'parking_availability_query': {
            '주차': 'parking_lot',
            '주차 공간': 'parking_lot',
            '주차장 이용': 'parking_lot',
            '주차장 현황': 'parking_lot',
            '주차할 곳': 'parking_lot',
            '주차할 공간': 'parking_lot',
        },
    }

    matching_intents = [i for i in keyword_to_slot_map if i in intent_list]

    for match_intent in matching_intents:
        for keyword, slot_name in keyword_to_slot_map[match_intent].items():
            if keyword in question or keyword in question_no_space:
                # 슬롯이 이미 존재하는지 확인
                if slot_name not in slots:
                    slots[slot_name] = []

                # 리스트가 아닌 경우 리스트로 변환
                if not isinstance(slots[slot_name], list):
                    slots[slot_name] = [slots[slot_name]] if slots[slot_name] else []

                # 키워드가 이미 리스트에 없는 경우에만 추가
                if keyword not in slots[slot_name]:
                    slots[slot_name].append(keyword)

    # Special rule for airport_congestion_prediction - 복잡한 혼잡도 쿼리 파싱
    if 'airport_congestion_prediction' in intent_list:
        congestion_slots = _parse_congestion_query(question, question_no_space)
        if congestion_slots:
            slots.update(congestion_slots)

    return slots

def _parse_congestion_query(question: str, question_no_space: str) -> dict:
    """혼잡도 예측 쿼리를 파싱하여 구조화된 슬롯 반환"""
    congestion_slots = {}
    
    # 하루 전체/일일 혼잡도 키워드
    daily_keywords = ['하루', '일일', '전체', '하루 전체', '오늘 전체', '하루종일', '종일']
    is_daily = any(keyword in question or keyword in question_no_space for keyword in daily_keywords)
    
    if is_daily:
        congestion_slots['is_daily'] = True
        congestion_slots['time_range'] = '합계'
    else:
        congestion_slots['is_daily'] = False
    
    # 날짜 추출
    if '내일' in question or '내일' in question_no_space:
        congestion_slots['date_type'] = 'tomorrow'
    elif '오늘' in question or '오늘' in question_no_space:
        congestion_slots['date_type'] = 'today'
    else:
        # 특정 날짜 패턴 (예: 12월 25일)
        date_pattern = re.search(r'(\d{1,2})월\s*(\d{1,2})일', question)
        if date_pattern:
            congestion_slots['date_type'] = 'unsupported'
            congestion_slots['specific_date'] = date_pattern.group(0)
        else:
            congestion_slots['date_type'] = 'today'  # 기본값
    
    # 터미널 정보는 이미 위에서 처리됨 (기존 Rule 1 사용)
    
    # 구역 정보 (입국장/출국장 + 알파벳/숫자)
    area_patterns = [
        r'(입국장)\s*([A-Z0-9]+)',     # "입국장A", "입국장1" 
        r'(출국장)\s*([A-Z0-9]+)',     # "출국장B", "출국장2"
        r'(입국장)', # 단순히 "입국장"만
        r'(출국장)', # 단순히 "출국장"만
    ]
    
    areas = []
    for pattern in area_patterns:
        matches = re.findall(pattern, question) or re.findall(pattern, question_no_space)
        for match in matches:
            if isinstance(match, tuple) and len(match) == 2:
                areas.append(match[0] + match[1])  # "입국장A"
            elif isinstance(match, str):
                areas.append(match)  # "입국장"
    
    if areas:
        congestion_slots['areas'] = list(set(areas))  # 중복 제거
    
    # 시간 정보
    if not is_daily:
        time_patterns = [
            r'(\d{1,2})시',           # "15시", "3시"
            r'오전\s*(\d{1,2})시',    # "오전 10시"
            r'오후\s*(\d{1,2})시',    # "오후 3시"
        ]
        
        for pattern in time_patterns:
            match = re.search(pattern, question)
            if match:
                hour = int(match.group(1))
                if '오후' in match.group(0) and hour != 12:
                    hour += 12
                elif '오전' in match.group(0) and hour == 12:
                    hour = 0
                congestion_slots['hour'] = hour
                break
    
    # 혼잡도 관련 키워드
    congestion_keywords = ['혼잡도', '혼잡', '붐비', '사람', '대기시간', '소요시간']
    found_keywords = [kw for kw in congestion_keywords if kw in question or kw in question_no_space]
    if found_keywords:
        congestion_slots['congestion_keywords'] = found_keywords
    
    return congestion_slots

def create_full_slot_dataset(input_file, output_file):
    print(f"Reading from {input_file}...")
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8-sig') as infile, \
         open(output_file, 'w', encoding='utf-8-sig', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = ['question', 'slots', 'intent_list']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            question = row['question']
            intent_list = row['intent_list']
            
            slots = extract_slots(question, intent_list)
            
            writer.writerow({
                'intent_list': intent_list,
                'question': question,
                'slots': json.dumps(slots, ensure_ascii=False)
            })
            processed_count += 1
    print(f"Successfully processed {processed_count} lines and created {output_file}")

if __name__ == '__main__':
    input_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/data/intent_dataset_cleaned.csv'
    output_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/data/intent_slot_dataset_cleaned.csv'
    create_full_slot_dataset(input_csv, output_csv)

    input_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/data/keyword_boost.csv'
    output_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/data/keyword_boost_slot.csv'
    create_full_slot_dataset(input_csv, output_csv)