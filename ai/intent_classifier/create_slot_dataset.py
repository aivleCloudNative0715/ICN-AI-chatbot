import csv
import json
import re
import pandas as pd

def load_keywords(keyword_file):
    """Loads keywords from the CSV file into a dictionary."""
    try:
        # Read without header, assign column names manually
        keyword_df = pd.read_csv(keyword_file, header=None, names=['intent', 'question'])
        keywords_by_intent = {}
        for intent, group in keyword_df.groupby('intent'):
            keywords_by_intent[intent] = group['question'].tolist()
        return keywords_by_intent
    except FileNotFoundError:
        print(f"Error: Keyword file not found at {keyword_file}")
        return {}

def remove_korean_josa(text):
    return re.sub(r'([A-Z]{3})([은는이가에서의]*)', r'\1', text)

def extract_slots(question, intent, keywords_by_intent):
    slots = {}

    # Rule 1: Terminal (T1, T2)
    t1_keywords = ['1터미널', 'T1', '제1터미널', '제 1 터미널', '일 터미널', '터미널 1', '터미널 일', '제일터미널', '일 터미널']
    t2_keywords = ['2터미널', 'T2', '제2터미널', '제 2 터미널', '이 터미널', '터미널 2', '터미널 이', '제이터미널', '이 터미널']
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
    flight_status_keywords = ['지연', '결항', '취소', '도착', '출발', '연착', '정시']
    for status in flight_status_keywords:
        if status in question:
            slots['flight_status'] = status
            break

    # NEW Rule for extracting origin and destination from expressions like "인천발 파리행"
    if intent == 'flight_info':
        # e.g., 인천발 파리행
        location_pattern = re.findall(r'([가-힣]+)[발|출발|에서]\s*([가-힣]+)[행|도착|까지|가는]', question)
        if location_pattern:
            slots['origin'] = location_pattern[0][0]
            slots['destination'] = location_pattern[0][1]
        else:
            # Separate match for origin
            origin_match = re.search(r'([가-힣]+)[발|출발|에서]', question)
            if origin_match:
                slots['origin'] = origin_match.group(1)
            # Separate match for destination
            destination_match = re.search(r'([가-힣]+)[행|도착|까지|가는]', question)
            if destination_match:
                slots['destination'] = destination_match.group(1)


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

    if intent == 'airline_info_query':
        if re.search(r'로고\s*링크|로고\s*이미지|로고', question):
            slots['topic'] = ['로고 이미지' if '이미지' in question else '로고 링크' if '링크' in question else '로고']
        elif '이미지' in question:
            slots['topic'] = ['이미지']

    # Rule 6: Facility Name (Enhanced with keywords)
    base_facilities = [
        '약국', '은행', '환전소', '수유실', '유아휴게실', '기도실', '면세점', '흡연실', '식당', '카페', '편의점',
        '로밍센터', '셔틀버스', '택배', '병원', '의료', '안내 데스크', '여권민원실', '병무청', '검역장', 'ATM기',
        '밥집', '의료관광 안내센터', '패스트푸드점', '한식', '정형외과', '내과', '피부과', '치과', '비즈니스센터',
        '수하물보관소', '우체국', '라운지', '여객터미널', '렌터카', '모노레일', '전망대', '샤워실', '찜질방',
        '이발소', '미용실', '호텔', '에스컬레이터', '엘리베이터', '에어트레인', '스카이워크', '자동체크인', '출입국관리소',
        '의료센터', '헬스케어센터', 'VR체험존', '키즈존', '문화센터', '전시관', '공항철도', '지하철역', 'KTX역',
        '공항버스정류장', '택시승강장', '주차장', '주차타워', '공항 안내소', '어린이 놀이시설','해외감염병신고센터',
        '환승장', '환승호텔', 'AED', 'ATM'
    ]

    facility_keywords = keywords_by_intent.get('facility_guide', [])
    # Combine, remove duplicates, and sort by length to match longer phrases first
    facilities = sorted(list(set(base_facilities + facility_keywords)), key=len, reverse=True)
    for facility in facilities:
        if facility in question:
            slots['facility_name'] = facility
            break

    # 출발/도착 유형
    if '국내선' in question:
        slots['departure_type'] = '국내선' if '타고' in question or '에서' in question else None
        slots['arrival_type'] = '국제선' if '으로' in question or '가야' in question else None
    if '국제선' in question:
        slots['departure_type'] = '국제선' if '타고' in question or '에서' in question else slots.get('departure_type')
        slots['arrival_type'] = '국내선' if '으로' in question or '가야' in question else slots.get('arrival_type')

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

    # NEW Rule for Airport Name
    if intent == 'airport_info':
        # Case 1: "인천공항", "김해 국제공항" 등 공항 이름 추출
        airport_name_match = re.search(r'([가-힣A-Za-z\s]+(공항|국제공항))', question)
        if airport_name_match:
            airport_name = airport_name_match.group(1).strip()
            if len(airport_name) > 3 and '공항' in airport_name:
                slots['airport_name'] = airport_name

        # Case 2: IATA 3-letter airport code (e.g., CDG, JFK, HND)
        cleaned = remove_korean_josa(question)
        iata_code_match = re.findall(r'\b([A-Z]{3})\b', cleaned)
        if iata_code_match:
            slots['airport_code'] = list(set(iata_code_match))  # e.g., ['CDG'

    # NEW Rule for Weather Topic
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
            slots['weather_topic'] = found_weather_topics

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

    vehicle_type_keywords = {
        '대형': ['대형차', '대형 차량', '버스', '트럭'],
        '소형': ['소형차', '소형 차량'],
        '장애인': ['장애인 차량', '장애인 주차']
    }
    for v_type, keywords in vehicle_type_keywords.items():
        if any(keyword in question for keyword in keywords):
            slots['vehicle_type'] = v_type
            break

    # NEW Rule for Parking Walk Time
    if intent == 'parking_walk_time_info':
        destination_match = re.search(r'(1터미널|2터미널|T1|T2|출국장|입국장)', question)
        if destination_match:
            slots['destination'] = destination_match.group(1)

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

    # NEW Rule for Parking Availability Query
    if intent == 'parking_availability_query':
        # Terminal
        if '1터미널' in question or 'T1' in question:
            slots['terminal'] = 'T1'
        elif '2터미널' in question or 'T2' in question:
            slots['terminal'] = 'T2'

        # Parking Lot (general)
        if '주차타워' in question:
            slots['parking_lot'] = '주차타워'
        elif '주차장' in question:
            slots['parking_lot'] = '주차장'

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
        
        # Availability Status
        availability_keywords = ['만차', '혼잡', '여유', '가능', '비어있', '꽉 찼']
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
        
        for item in items:
            if item in question:
                slots['item'] = item
                break

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
            slots['baggage_issue'] = 'lost'
        elif '파손' in question:
            slots['baggage_issue'] = 'damaged'
        elif '늦어' in question:
            slots['baggage_issue'] = 'delayed'
        
        # Baggage type (general, special, excess)
        if any(keyword in question for keyword in ['일반 수하물', '일반 짐']):
            slots['baggage_type'] = 'general'
        elif any(keyword in question for keyword in ['유모차', '휠체어', '특수 수하물', '스포츠 장비', '악기']):
            slots['baggage_type'] = 'special'
        elif any(keyword in question for keyword in ['초과 수하물', '추가 짐', '무게 초과']):
            slots['baggage_type'] = 'excess'

        luggage_terms = ['수하물', '짐', '가방']
        found_luggage_terms = [term for term in luggage_terms if term in question]
        if found_luggage_terms:
            slots['luggage_term'] = found_luggage_terms

    # NEW Rule: Transfer-related topics
    if intent == 'transfer_info':
        transfer_keywords = {
            'stopover': ['스탑오버', 'stopover'],
            'layover_program': ['관광 프로그램', '투어 프로그램', '트랜짓 투어', '경유 관광'],
            'shuttle': ['셔틀', '공항 셔틀', '셔틀버스'],
            'LAGs': ['LAGs', '액체류', '젤류', '에어로졸', '액체 젤', '보안검색 액체'],
            'health_declaration': ['건강상태질문서', '건강 상태 질문서', '건강 상태 신고서'],
            'customs': ['전자세관신고', '세관 신고', '세관 심사', '관세 신고'],
            'airlines': ['항공사', '항공사 목록', '제휴 항공사']
        }

        for key, keywords in transfer_keywords.items():
            for keyword in keywords:
                if keyword in question:
                    slots['transfer_topic'] = key
                    break
            if 'transfer_topic' in slots:
                break

    if intent == 'departure_policy_info':
        policy_keywords = {
            'passport_type': ['관용여권', '일반여권', '전자여권', '긴급여권'],
            'person_type': ['국외체류자', '외국인', '내국인', '병역대상자'],
            'item': ['귀금속', '화폐', '미술품', '문화재'],
            'facility_name': ['여권민원센터', '감정관', '민원실'],
            'organization': ['외교부', '병무청', '출입국관리소'],
            'document': ['비행기표', '여권', '비자', '서류'],
            'topic': ['수수료', '반납', '신청', '확인', '환급', '위치', '운영시간', '서류']
        }

        for slot_name, keywords in policy_keywords.items():
            found = []
            for keyword in keywords:
                if keyword in question:
                    found.append(keyword)

            if found:
                slots[slot_name] = found if len(found) > 1 else found[0]

    # Rule 8: Topic Slot from Keywords
    topic_keywords = keywords_by_intent.get(intent, [])
    found_topics = []
    # Sort by length to match longer keywords first
    for keyword in sorted(topic_keywords, key=len, reverse=True):
        if keyword in question:
            # Avoid adding a topic that is a substring of an already found topic
            if not any(keyword in found_topic for found_topic in found_topics):
                 found_topics.append(keyword)

    if found_topics:
        slots['topic'] = found_topics

    return slots

def create_full_slot_dataset(input_file, output_file, keyword_file):
    print(f"Loading keywords from {keyword_file}...")
    keywords_by_intent = load_keywords(keyword_file)

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
            
            slots = extract_slots(question, intent, keywords_by_intent)
            
            writer.writerow({
                'intent': intent,
                'question': question,
                'slots': json.dumps(slots, ensure_ascii=False)
            })
            processed_count += 1
    print(f"Successfully processed {processed_count} lines and created {output_file}")

if __name__ == '__main__':
    input_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/Old_data/intent_dataset.csv'
    output_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/intent_slot_dataset.csv'
    keyword_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/Old_data/keyword_boost.csv'
    create_full_slot_dataset(input_csv, output_csv, keyword_csv)

    input_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/Old_data/keyword_boost.csv'
    output_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/keyword_boost_slot.csv'
    keyword_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/Old_data/keyword_boost.csv'
    create_full_slot_dataset(input_csv, output_csv, keyword_csv)
