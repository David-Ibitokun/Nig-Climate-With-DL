"""
Use the FAO CSV content (embedded) to replace crop yields in the repo,
then rebuild project datasets so models use the real yields.

This script was generated to run inside the repository environment.
"""
from io import StringIO
import os
import pandas as pd
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_PR = os.path.join(ROOT, 'ProjectReadyData')
OUT_CROP = os.path.join(ROOT, 'project_data', 'crop_yields')
os.makedirs(OUT_PR, exist_ok=True)
os.makedirs(OUT_CROP, exist_ok=True)

FAO_RAW = r"""
Domain Code,Domain,Area Code (FAO),Area,Element Code,Element,Item Code (FAO),Item,Year Code,Year,Unit,Value,Flag,Flag Description,Note
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",1990,1990,kg/ha,11653.3,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",1991,1991,kg/ha,10193.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",1992,1992,kg/ha,10593.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",1993,1993,kg/ha,10593.5,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",1994,1994,kg/ha,10592.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",1995,1995,kg/ha,10667.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",1996,1996,kg/ha,10664.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",1997,1997,kg/ha,11881.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",1998,1998,kg/ha,10746.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",1999,1999,kg/ha,9599.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2000,2000,kg/ha,9700,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2001,2001,kg/ha,9601.2,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2002,2002,kg/ha,9901.3,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2003,2003,kg/ha,10402.3,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2004,2004,kg/ha,11001.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2005,2005,kg/ha,10990.2,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2006,2006,kg/ha,12000.3,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2007,2007,kg/ha,11202.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2008,2008,kg/ha,11800.4,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2009,2009,kg/ha,11767.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2010,2010,kg/ha,12215.5,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2011,2011,kg/ha,11210.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2012,2012,kg/ha,7958.5,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2013,2013,kg/ha,7032.3,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2014,2014,kg/ha,7655.4,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2015,2015,kg/ha,7265.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2016,2016,kg/ha,9076.3,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2017,2017,kg/ha,9073.7,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2018,2018,kg/ha,9403.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2019,2019,kg/ha,5827.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2020,2020,kg/ha,5779.5,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2021,2021,kg/ha,5835.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2022,2022,kg/ha,6014.5,E,Estimated value,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,125,"Cassava, fresh",2023,2023,kg/ha,6345.9,E,Estimated value,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),1990,1990,kg/ha,1130.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),1991,1991,kg/ha,1129.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),1992,1992,kg/ha,1118.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),1993,1993,kg/ha,1184.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),1994,1994,kg/ha,1272,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),1995,1995,kg/ha,1266.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),1996,1996,kg/ha,1326.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),1997,1997,kg/ha,1251,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),1998,1998,kg/ha,1320,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),1999,1999,kg/ha,1599.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2000,2000,kg/ha,1300.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2001,2001,kg/ha,1399.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2002,2002,kg/ha,1489.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2003,2003,kg/ha,1499.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2004,2004,kg/ha,1600.2,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2005,2005,kg/ha,1659.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2006,2006,kg/ha,1818.2,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2007,2007,kg/ha,1704.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2008,2008,kg/ha,1957.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2009,2009,kg/ha,2196.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2010,2010,kg/ha,1850.2,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2011,2011,kg/ha,1627.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2012,2012,kg/ha,1511.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2013,2013,kg/ha,1461.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2014,2014,kg/ha,1700.2,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2015,2015,kg/ha,1718.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2016,2016,kg/ha,1783.7,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2017,2017,kg/ha,1664.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2018,2018,kg/ha,1671.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2019,2019,kg/ha,1857.2,E,Estimated value,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2020,2020,kg/ha,2050.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2021,2021,kg/ha,2053.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2022,2022,kg/ha,2232.6,X,Figure from external organization,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,56,Maize (corn),2023,2023,kg/ha,1939.1,X,Figure from external organization,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,1990,1990,kg/ha,10677.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,1991,1991,kg/ha,10345.3,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,1992,1992,kg/ha,11348.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,1993,1993,kg/ha,11349.4,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,1994,1994,kg/ha,11399.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,1995,1995,kg/ha,10773.4,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,1996,1996,kg/ha,10681.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,1997,1997,kg/ha,11047.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,1998,1998,kg/ha,9435.4,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,1999,1999,kg/ha,9901.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2000,2000,kg/ha,9898.4,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2001,2001,kg/ha,9799,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2002,2002,kg/ha,9989.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2003,2003,kg/ha,10501.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2004,2004,kg/ha,11200.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2005,2005,kg/ha,11498.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2006,2006,kg/ha,12098.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2007,2007,kg/ha,9969.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2008,2008,kg/ha,11499.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2009,2009,kg/ha,10479.7,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2010,2010,kg/ha,13010.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2011,2011,kg/ha,7403.7,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2012,2012,kg/ha,7201.3,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2013,2013,kg/ha,7000.1,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2014,2014,kg/ha,8194.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2015,2015,kg/ha,8182.7,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2016,2016,kg/ha,9159.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2017,2017,kg/ha,9385.6,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2018,2018,kg/ha,10199.9,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2019,2019,kg/ha,7806.8,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2020,2020,kg/ha,7904.7,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2021,2021,kg/ha,7986.5,A,Official figure,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2022,2022,kg/ha,8182.9,E,Estimated value,
QCL,Crops and livestock products,159,Nigeria,5412,Yield,137,Yams,2023,2023,kg/ha,8203.4,E,Estimated value,
"""

# Parse embedded CSV
sio = StringIO(FAO_RAW)
df = pd.read_csv(sio)

# Filter for our items
items_map = {
    'Maize (corn)': 'Maize',
    'Cassava, fresh': 'Cassava',
    'Yams': 'Yam',
    'Yam': 'Yam'
}

# Keep rows where Item matches any of these keys
mask = df['Item'].isin(items_map.keys())
df_sel = df[mask].copy()
# Normalize
df_sel['Crop'] = df_sel['Item'].map(items_map)
# Ensure Year and Value
df_sel = df_sel[['Year', 'Crop', 'Value']].rename(columns={'Value':'Yield_kg_per_ha'})
# Some Value may be strings; convert
df_sel['Yield_kg_per_ha'] = pd.to_numeric(df_sel['Yield_kg_per_ha'], errors='coerce')

# Save to ProjectReadyData and project_data/crop_yields
out_pr = os.path.join(OUT_PR, 'nigeria_crop_yields.csv')
df_sel.to_csv(out_pr, index=False)
print('Wrote', out_pr)

out_proj = os.path.join(OUT_CROP, 'nigeria_crop_yields_1990_2023_fromFAO.csv')
df_sel.to_csv(out_proj, index=False)
print('Wrote', out_proj)

# Now regenerate model datasets by calling the collector
try:
    from scripts.download_zonal_data import NigeriaProjectDataCollector
    collector = NigeriaProjectDataCollector()
    print('Running create_project_dataset() to regenerate models...')
    res = collector.create_project_dataset()
    # Report shapes and file sizes
    models_dir = os.path.join(ROOT, 'project_data', 'models')
    files = ['fnn_dataset.csv','lstm_sequences.npy','lstm_targets.npy']
    for f in files:
        path = os.path.join(models_dir,f)
        if os.path.exists(path):
            size = os.path.getsize(path)
            print(f'{f}: exists, size={size} bytes')
            # If numpy arrays, load shape
            if f.endswith('.npy'):
                try:
                    arr = np.load(path)
                    print('  shape:', arr.shape)
                except Exception as e:
                    print('  could not load .npy:', e)
        else:
            print(f'{f}: MISSING')
except Exception as e:
    print('Failed to regenerate models:', e)

print('Done')
