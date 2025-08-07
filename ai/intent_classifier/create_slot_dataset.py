import csv
import json
import re
from collections import defaultdict

from konlpy.tag import Okt

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

def extract_slots(question, intent):
    slots = defaultdict(list)

    # Rule 1: Terminal (T1, T2)
    t1_keywords = ['1터미널', 'T1', '제1터미널', '제 1 터미널', '일 터미널', '터미널 1', '터미널 일', '제일터미널']
    t2_keywords = ['2터미널', 'T2', '제2터미널', '제 2 터미널', '이 터미널', '터미널 2', '터미널 이', '제이터미널']
    if any(keyword in question for keyword in t1_keywords):
        slots['terminal'] = 'T1'
    elif any(keyword in question for keyword in t2_keywords):
        slots['terminal'] = 'T2'

    # Rule 2: Area (입국/출국 관련)
    arrival_keywords = ['입국장', '입국 게이트', '도착층', '도착 게이트']
    departure_keywords = ['출국장', '출국 게이트', '출발층', '탑승구']
    
    is_arrival = any(keyword in question for keyword in arrival_keywords)
    is_departure = any(keyword in question for keyword in departure_keywords)

    if is_arrival and not is_departure:
        slots['area'] = 'arrival'
    elif is_departure and not is_arrival:
        slots['area'] = 'departure'
    # If both are present, the intent might be the tie-breaker
    elif 'arrival' in intent:
        slots['area'] = 'arrival'
    elif 'departure' in intent:
        slots['area'] = 'departure'

    # Rule 3: Gate
    gate_match = re.search(r'([A-Z]{1,2}|[0-9]{1,3})\s?[번]?\s?(게이트|출국장|입국장|탑승구)', question)
    if gate_match and gate_match.group(1):
        slots['gate'] = gate_match.group(1).strip()

    # Rule 4: Airline Flight
    flight_match = re.search(r'\b([A-Z0-9]{2}\d{1,4})(?:편)?', question, re.IGNORECASE)
    if flight_match:
        slots['airline_flight'] = flight_match.group(1).upper()

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
    if any(keyword in question for keyword in ['항공편', '비행기', '항공기', '항공권', '운항']) or 'airline_flight' in slots:
        if '도착' in question:
            slots['flight_status'] = '도착'
        elif '출발' in question:
            slots['flight_status'] = '출발'

    for status in flight_status_keywords:
        if status in question:
            slots['flight_status'] = status
            break

    airports = ['인천', '김포', '제주', '상하이', '도쿄', '뉴욕', '파리', '런던', '프랑크푸르트', '방콕', 'LA', '시드니', '헬싱키',
                '후쿠오카', '쿠알라룸푸르', '마닐라', '모스크바', '아부다비', '밴쿠버', '달라스', '삿포로', '런던 히드로', '싱가포르']

    for ap in airports:
        if f'{ap}발' in question:
            slots['departure_airport'] = ap
        if f'{ap}행' in question:
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
        if airline in question:
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

    if intent == 'facility_guide':
        for facility in base_facilities:
            if facility in question:
                slots['facility_name'] = facility
                break

    if '국내선' in question or '국제선' in question:
        if '국내선' in question:
            if re.search(r'(국내선.*(에서|출발|타고|발))', question):
                slots['departure_type'] = '국내선'
            if re.search(r'(국내선.*(으로|가는|도착|가야|향해))', question):
                slots['arrival_type'] = '국내선'

        if '국제선' in question:
            if re.search(r'(국제선.*(에서|출발|타고|발))', question):
                slots['departure_type'] = '국제선'
            if re.search(r'(국제선.*(으로|가는|도착|가야|향해))', question):
                slots['arrival_type'] = '국제선'

    # 시설
    facility_keywords = ['이동통로', '탑승동', '수속']
    for keyword in facility_keywords:
        if keyword in question:
            slots['location_keyword'] = keyword
            break

    # Rule 7: Date/Time (Enhanced)
    date_match_md = re.search(r'(\d{1,2}월\s?\d{1,2}일)', question)
    if date_match_md:
        slots['date'] = date_match_md.group(1).strip()
    elif '오늘' in question:
        slots['date'] = '오늘'
    elif '내일' in question:
        slots['date'] = '내일'
    elif '모레' in question:
        slots['date'] = '모레'
    elif '어제' in question:
        slots['date'] = '어제'

    day_of_week_keywords = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    for day in day_of_week_keywords:
        if day in question:
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
            if vague_time in question:
                slots['vague_time'] = vague_time
                extracted_vague_time = True
                break

        if not extracted_vague_time:
            for period in time_period_keywords:
                if period in question:
                    if '시' not in question:
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
            if word in question:
                slots['season'] = word
                break

    if intent == 'airport_info':
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

    if intent == 'airport_weather_current':
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

    # NEW Rule for Parking Slots (General)
    parking_lot_match = re.search(r'(P[1-5]|장기주차장|단기주차장|화물터미널 주차장|주차타워|주차장)', question)
    if parking_lot_match:
        slots['parking_lot'] = parking_lot_match.group(1)

    parking_type_keywords = {
        '장기': ['장기주차장', '장기 주차'],
        '단기': ['단기주차장', '단기 주차'],
        '화물': ['화물터미널 주차장', '화물 주차'],
        '예약': ['예약 주차장', '예약 주차']
    }
    for p_type, keywords in parking_type_keywords.items():
        if any(keyword in question for keyword in keywords):
            slots['parking_type'] = p_type
            break


    # NEW Rule for Parking Fee Info
    if intent == 'parking_fee_info':
        duration_match = re.search(r'(\d+)\s*(시간|분|일)', question)
        if duration_match:
            slots['parking_duration_value'] = duration_match.group(1)
            slots['parking_duration_unit'] = duration_match.group(2)
        
        payment_method_keywords = ['카드', '현금', '하이패스', '모바일', '간편결제']
        for method in payment_method_keywords:
            if method in question:
                slots['payment_method'] = method
                break


    # Unified Rule for Parking Availability and Congestion
    if intent in ['parking_availability_query', 'parking_congestion_prediction']:
        # Terminal
        if '1터미널' in question or 'T1' in question:
            slots['terminal'] = 'T1'
        elif '2터미널' in question or 'T2' in question:
            slots['terminal'] = 'T2'

        # Parking Type
        if '단기 주차장' in question or '단기주차장' in question:
            slots['parking_type'] = '단기'
        elif '장기 주차장' in question or '장기주차장' in question:
            slots['parking_type'] = '장기'

        # Parking Area (specific sections)
        parking_area_keywords = [
            'P1', 'P2', 'P3', 'P4', 'P5', '지하 1층', '지하 2층', '지상 1층', '지상 2층',
            '지하', '지상', '동편', '서편', '화물터미널 주차장'
        ]
        for area in sorted(parking_area_keywords, key=len, reverse=True):
            if area in question:
                slots['parking_area'] = area
                break
        
        # Availability Status (for availability query)
        if intent == 'parking_availability_query':
            availability_keywords = ['만차', '혼잡', '여유', '가능', '비어있', '꽉 찼', '주차 가능', '만차 여부', '자리 있'
                                     '자리 없', '자리 확인']
            for status in availability_keywords:
                if status in question:
                    slots['availability_status'] = status
                    break

        # Date/Time (re-using existing rules for consistency)
        date_match_md = re.search(r'(\d{1,2}월\s?\d{1,2}일)', question)
        if date_match_md:
            slots['date'] = date_match_md.group(1).strip()
        elif '오늘' in question:
            slots['date'] = '오늘'
        elif '내일' in question:
            slots['date'] = '내일'
        elif '모레' in question:
            slots['date'] = '모레'
        elif '어제' in question:
            slots['date'] = '어제'

        day_of_week_keywords = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
        for day in day_of_week_keywords:
            if day in question:
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
                if vague_time in question:
                    slots['time'] = vague_time
                    break
            else:
                for period in time_period_keywords:
                    if period in question and '시' not in question:
                        slots['time'] = period
                        break

    # NEW Rule for Baggage Rule
    if intent == 'baggage_rule_query':
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
        found_luggage_terms = [term for term in luggage_terms if term in question]
        if found_luggage_terms:
            slots['luggage_term'] = found_luggage_terms

        for b_type, keywords in baggage_types.items():
            if any(keyword in question for keyword in keywords):
                slots['baggage_type'] = b_type
                break

        for r_type, keywords in rule_types.items():
            if any(keyword in question for keyword in keywords):
                slots['rule_type'] = r_type
                break

        found_items = [item for item in items if item in question]
        if found_items:
            slots['item'] = found_items

    # NEW Rule: Self Bag Drop 여부 확인
    if intent == 'baggage_rule_query':
        self_bag_drop_keywords = [
            '셀프 백드랍', '셀프 백드롭', 'self bag drop', 'self baggage drop', '셀프 수하물 위탁'
        ]
        for keyword in self_bag_drop_keywords:
            if keyword in question.lower():
                slots['self_bag_drop'] = True
                break

    # NEW Rule for Baggage Claim Info
    if intent == 'baggage_claim_info':
        # Baggage belt number
        belt_match = re.search(r'(\d+)\s?[번]?\s?(수취대|컨베이어 벨트)', question)
        if belt_match:
            slots['baggage_belt_number'] = belt_match.group(1).strip()

        # Baggage issue (lost, damaged, delayed)
        if any(keyword in question for keyword in ['안 나왔', '분실', '못 찾']):
            slots['baggage_issue'] = '분실'
        elif '파손' in question:
            slots['baggage_issue'] = '파손'
        elif '늦어' in question:
            slots['baggage_issue'] = '지연'
        
        # Baggage type (general, special, excess)
        if any(keyword in question for keyword in ['일반 수하물', '일반 짐']):
            slots['baggage_type'] = 'general'
        elif any(keyword in question for keyword in ['유모차', '휠체어', '특수 수하물', '스포츠 장비', '악기']):
            slots['baggage_type'] = 'special'
        elif any(keyword in question for keyword in ['초과 수하물', '추가 짐', '무게 초과']):
            slots['baggage_type'] = 'excess'

        luggage_keywords = ['짐', '수하물', '캐리어', '가방', '화물']

        for term in luggage_keywords:
            if term in question:
                slots['luggage_term'] = [term]
                break

    # NEW Rule: Transfer-related topics
    if intent == 'transfer_info':
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
                if keyword in question:
                    slots['transfer_topic'] = key
                    break
            if 'transfer_topic' in slots:
                break

    if intent == 'immigration_policy':
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
                if keyword in question:
                    found.append(keyword)

            if found:
                slots[slot_name] = found if len(found) > 1 else found[0]

    if intent == 'parking_walk_time_info':
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
        'airport_congestion_prediction': {
            '입국장 혼잡 예측': 'congestion_topic',
            '입국장': 'congestion_topic',
            '입국장 예측': 'congestion_topic',
            '입국장 혼잡도': 'congestion_topic',
            '입국장 미래': 'congestion_topic',
            '입국장 승객': 'congestion_topic',
            '예상 소요시간': 'congestion_topic',
            '출국장': 'congestion_topic',
            '출국장 예측': 'congestion_topic',
            '출국장 혼잡도': 'congestion_topic',
            '출국장 미래': 'congestion_topic',
            '출국장 승객': 'congestion_topic',
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


    if intent in keyword_to_slot_map:
        for keyword, slot_name in keyword_to_slot_map[intent].items():
            if keyword in question:
                if isinstance(slots[slot_name], list):
                    slots[slot_name].append(keyword)
                else:
                    slots[slot_name] = [slots[slot_name], keyword]

    return slots

def create_full_slot_dataset(input_file, output_file):
    print(f"Reading from {input_file}...")
    processed_count = 0
    with open(input_file, 'r', encoding='utf-8-sig') as infile, \
         open(output_file, 'w', encoding='utf-8-sig', newline='') as outfile:
        
        reader = csv.DictReader(infile)
        fieldnames = ['intent', 'question', 'slots']
        writer = csv.DictWriter(outfile, fieldnames=fieldnames)
        writer.writeheader()

        for row in reader:
            question = row['question']
            intent = row['intent']
            
            slots = extract_slots(question, intent)
            
            writer.writerow({
                'intent': intent,
                'question': question,
                'slots': json.dumps(slots, ensure_ascii=False)
            })
            processed_count += 1
    print(f"Successfully processed {processed_count} lines and created {output_file}")

if __name__ == '__main__':
    input_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/data/intent_dataset.csv'
    output_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/data/intent_slot_dataset.csv'
    create_full_slot_dataset(input_csv, output_csv)

    input_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/data/keyword_boost.csv'
    output_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/data/keyword_boost_slot.csv'
    create_full_slot_dataset(input_csv, output_csv)