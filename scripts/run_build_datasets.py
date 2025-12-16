from scripts.download_zonal_data import NigeriaProjectDataCollector
import os

c = NigeriaProjectDataCollector()
print('1) Generating crop yields (FAOSTAT fallback)')
c.get_crop_yield_data()

print('\n2) Refreshing food security indicators')
c.get_food_security_data()

print('\n3) Creating project datasets (FNN & LSTM)')
res = c.create_project_dataset()

print('\nDone. Created files:')
for root, dirs, files in os.walk('project_data'):
    for f in files:
        print('-', os.path.join(root, f))

if res is None:
    print('\nERROR: Dataset creation returned None')
else:
    print('\nDatasets keys:', list(res.keys()))
