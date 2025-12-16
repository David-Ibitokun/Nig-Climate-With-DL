import os
import json
import numpy as np
base='project_data/models'
files=['fnn_dataset.csv','lstm_sequences.npy','lstm_targets.npy']
out={}
for f in files:
    p=os.path.join(base,f)
    if os.path.exists(p):
        info={'size':os.path.getsize(p)}
        if f.endswith('.npy'):
            try:
                info['shape']=tuple(np.load(p).shape)
            except Exception as e:
                info['shape']=f'load_failed:{e}'
    else:
        info=None
    out[f]=info
print(json.dumps(out, indent=2))
