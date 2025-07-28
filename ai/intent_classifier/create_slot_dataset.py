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

def extract_slots(question, intent, keywords_by_intent):
    slots = {}

    # Rule 1: Terminal (T1, T2)
    t1_keywords = ['1터미널', 'T1', '제1터미널', '제 1 터미널']
    t2_keywords = ['2터미널', 'T2', '제2터미널', '제 2 터미널']
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
    flight_match = re.search(r'([A-Z]{2}\d{3,4})', question, re.IGNORECASE)
    if flight_match:
        slots['airline_flight'] = flight_match.group(1).upper()

    # Rule 5: Facility Name (Enhanced with keywords)
    base_facilities = ['약국', '은행', '환전소', '수유실', '유아휴게실', '기도실', '면세점', '흡연실', '식당', '카페', '편의점', '로밍센터', '셔틀버스', '택배', '병원', '의료', '안내 데스크', '여권민원실', '병무청']
    facility_keywords = keywords_by_intent.get('facility_guide', [])
    # Combine, remove duplicates, and sort by length to match longer phrases first
    facilities = sorted(list(set(base_facilities + facility_keywords)), key=len, reverse=True)
    for facility in facilities:
        if facility in question:
            slots['facility_name'] = facility
            break

    # Rule 6: Date/Time (Enhanced)
    # Specific dates: MM월 DD일
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

    # Days of the week
    day_of_week_keywords = ['월요일', '화요일', '수요일', '목요일', '금요일', '토요일', '일요일']
    for day in day_of_week_keywords:
        if day in question:
            slots['day_of_week'] = day
            break

    # Time: HH시 MM분, HH시, HH시 반
    time_match_hm = re.search(r'((오전|오후|아침|저녁|밤|새벽)?\s?\d{1,2}시\s?\d{1,2}분)', question)
    time_match_h = re.search(r'((오전|오후|아침|저녁|밤|새벽)?\s?\d{1,2}시(\s?반)?)', question)

    if time_match_hm:
        slots['time'] = time_match_hm.group(1).strip()
    elif time_match_h:
        slots['time'] = time_match_h.group(1).strip()
    elif '지금' in question or '현재' in question:
        slots['time'] = '현재'

    # New Rule 7: Topic Slot from Keywords
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
    keyword_csv = 'C:/Users/jenny/PycharmProjects/ICN-AI-chatbot/ai/intent_classifier/keyword_boost.csv'
    create_full_slot_dataset(input_csv, output_csv, keyword_csv)
