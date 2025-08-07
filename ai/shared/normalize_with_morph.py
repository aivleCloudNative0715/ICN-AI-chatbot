from konlpy.tag import Okt
import re

okt = Okt()


def normalize_with_morph(text: str) -> str:
    # 터미널 키워드 리스트
    t1_keywords = ['1터미널', 'T1', '제1터미널', '제 1 터미널', '일 터미널', '터미널 1', '터미널 일', '제일터미널']
    t2_keywords = ['2터미널', 'T2', '제2터미널', '제 2 터미널', '이 터미널', '터미널 2', '터미널 이', '제이터미널']

    # 1. 항공편명 패턴 정의: 'TW603'
    flight_pattern = re.compile(r'[A-Z]{2,3}\d{3,4}')

    # 2. 터미널 키워드 패턴 정의
    # '|'를 이용해 여러 키워드를 OR 조건으로 연결
    t1_pattern = re.compile('|'.join(re.escape(k) for k in t1_keywords))
    t2_pattern = re.compile('|'.join(re.escape(k) for k in t2_keywords))

    # 3. 전처리 적용
    processed_text = text

    # 터미널 키워드를 먼저 처리하여 정확도를 높임
    processed_text = t1_pattern.sub(' 제1터미널 ', processed_text)
    processed_text = t2_pattern.sub(' 제2터미널 ', processed_text)

    # 항공편명 처리
    processed_text = flight_pattern.sub(lambda m: m.group(0) + ' ', processed_text)

    # 4. 형태소 분석
    tokens = okt.morphs(processed_text, stem=False)

    # 중복 공백 제거 및 정리
    normalized_text = " ".join(tokens)
    return normalized_text.strip()