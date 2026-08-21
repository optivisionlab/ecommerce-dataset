import json
import pandas as pd

with open('/mnt/user-data/uploads/befood_restaurant_data.json', encoding='utf-8') as f:
    data = json.load(f)

df = pd.DataFrame(data)
df['city_name'] = df['city'].map({189: 'TP.HCM', 190: 'Hà Nội'})
df['rating'] = pd.to_numeric(df['rating'], errors='coerce')
df['review_count'] = pd.to_numeric(df['review_count'], errors='coerce')
df['latitude'] = pd.to_numeric(df['latitude'], errors='coerce')
df['longitude'] = pd.to_numeric(df['longitude'], errors='coerce')

# derive district from crawl_region e.g. "TP.HCM - Quan 1" -> "Quan 1"
df['district'] = df['crawl_region'].str.split(' - ').str[1]

df.to_pickle('/home/claude/viz/df.pkl')
print(df.shape)
print(df[['city_name','district']].drop_duplicates().shape)
print(df['merchant_category'].value_counts())
