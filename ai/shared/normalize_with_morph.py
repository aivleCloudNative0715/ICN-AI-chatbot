from konlpy.tag import Okt

# 형태소 분석기 객체
okt = Okt()

def normalize_with_morph(text: str) -> str:
    """
    형태소 분석을 통해 의미 단위로 잘게 쪼개고 띄어쓰기를 재정렬함
    예: '카페가고싶은데' -> '카페 가고 싶 은데'
    """
    tokens = okt.morphs(text, stem=False)  # 품사 보존 필요 시 pos() 사용
    return " ".join(tokens)