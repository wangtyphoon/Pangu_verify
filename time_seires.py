import pandas as pd
from datetime import datetime

df = pd.read_csv('valid_sample.csv')

# 假設 filename 欄位名稱為 'filename'
def extract_time(filename):
    # 取前10碼: YYYYMMDDHH
    return datetime.strptime(filename[:10], '%Y%m%d%H')

df['datetime'] = df['filename'].apply(extract_time)

print(df[['filename', 'datetime']].head())


df['next_time'] = df['datetime'] + pd.Timedelta(hours=6)