import pandas as pd
import requests
from io import StringIO

url='https://gml.noaa.gov/webdata/ccgg/trends/co2/co2_mm_mlo.txt'

print('Fetching URL...')
r = requests.get(url, timeout=30)
print('Status code:', r.status_code)

text = r.text
lines = text.splitlines()
print('Total lines fetched:', len(lines))
print('First 30 lines:')
for i,l in enumerate(lines[:30]):
    print(i+1, l[:200])

# find first non-comment line and show its tokens
for idx,l in enumerate(lines):
    if l.strip() and l.strip()[0].isdigit():
        print('\nFirst numeric data line at', idx+1)
        print('LINE:', repr(l))
        toks = l.split()
        print('TOK COUNT:', len(toks))
        print('TOKS SAMPLE:', toks[:12])
        break
# Try pandas read with comment
try:
    df = pd.read_csv(url, comment='#', delim_whitespace=True, header=None,
                     names=['year','month','decimal_date','average','interpolated','trend','days'], engine='python')
    print('\nPandas read with comment=# succeeded.')
    print('Parsed rows:', len(df))
    print(df.head(8).to_string())
except Exception as e:
    print('\nPandas read failed:', e)
    text2='\n'.join([l for l in lines if not l.strip().startswith('#') and l.strip()])
    df = pd.read_csv(StringIO(text2), delim_whitespace=True, header=None,
                     names=['year','month','decimal_date','average','interpolated','trend','days'])
    print('\nFallback parse rows:', len(df))
    print(df.head(8).to_string())

print('\nSample year values (unique sorted tail):')
print(sorted(df['year'].unique())[-15:])

# inspect types and first few year entries
print('\ndf.dtypes:\n', df.dtypes)
print('\nFirst 20 year values:', df['year'].iloc[:20].values)

# Filter 2010-2023 and show count
co2010 = df[pd.to_numeric(df['year'], errors='coerce').notnull()]
co2010['year'] = co2010['year'].astype(int)
co2010 = co2010[(co2010['year']>=2010)&(co2010['year']<=2023)]
print('\nFiltered 2010-2023 rows:', len(co2010))
print(co2010.tail(5).to_string())
