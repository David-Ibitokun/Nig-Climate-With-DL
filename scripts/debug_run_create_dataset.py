import traceback
from scripts.download_zonal_data import NigeriaProjectDataCollector
outpath='project_data/models/run_debug.txt'
print('Starting debug run; writing to', outpath)
with open(outpath,'w') as f:
    try:
        c=NigeriaProjectDataCollector()
        f.write('Collector created\n')
        print('Collector created')
        res=c.create_project_dataset()
        f.write('Result type: '+str(type(res))+'\n')
        print('Result type:', type(res))
        if res is None:
            f.write('Result is None\n')
            print('Result is None')
        else:
            f.write('Keys: '+str(list(res.keys()))+'\n')
            print('Keys:', list(res.keys()))
    except Exception:
        f.write('EXCEPTION:\n')
        traceback.print_exc(file=f)
        traceback.print_exc()
print('Done. Check', outpath)
