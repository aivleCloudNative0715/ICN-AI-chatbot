import pandas as pd
import glob

# 10개 CSV 파일 경로 패턴
file_pattern = "city_part_*.csv"

# 모든 CSV 파일 불러오기
csv_files = sorted(glob.glob(file_pattern))  # 정렬 필수: part_1, part_2, ...

# 데이터프레임 리스트로 읽기
dfs = [pd.read_csv(f) for f in csv_files]

# 하나로 합치기
merged_df = pd.concat(dfs, ignore_index=True)

# 결과 저장
merged_df.to_csv("city_merged.csv", index=False, encoding='utf-8-sig')

print(f"CSV 파일 {len(csv_files)}개 합쳐서 city_merged.csv 저장 완료!")
