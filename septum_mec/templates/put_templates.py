import expipe
import os
import os.path as op
import json
import quantities as pq

########################### DANGER DELETES ALL PAR.TEMPLATES #######################
# expipe.core.FirebaseBackend("").set('/templates', {})
# expipe.core.FirebaseBackend("").set('/templates_contents', {})
################################################################################

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

        template = {
            "identifier": name,
            "name": name,
        }
        print('Put ' + name)
        expipe.core.FirebaseBackend("/templates").set(name, template)
        expipe.core.FirebaseBackend("/templates_contents").set(name, result)
