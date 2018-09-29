import expipe
import os.path as op
import os
import json
import sys

if 'overwrite' in sys.argv:
    overwrite = True
else:
    overwrite = False


templates = expipe.core.FirebaseBackend("/templates").get()
for root, dirs, files in os.walk('templates'):
    for fname in files:
        if not fname.endswith('.json'):
            continue
        name = op.splitext(fname)[0]
        with open(op.join(root, fname), 'r') as infile:
            try:
                result = json.load(infile)
            except:
                print(fname)
                raise
        ########## EDIT ##########
        name = name.replace('mikkel_', '')
        fname = fname.replace('mikkel_', '')
        result.update({
            "identifier": name,
            "name": name,
        })
        #########################
        if not overwrite:
            print('The filename "' + fname + '" exists, set "ovewrite" to true.')
            print(result)
        else:
            print('Saving template "' + name + '" to "' + fname + '"')
            with open(op.join(root, fname), 'w') as outfile:
                json.dump(result, outfile,
                          sort_keys=True, indent=4)
