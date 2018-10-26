import expipe
import os
import os.path as op
import json
import quantities as pq

server = expipe.load_file_system(root='/home/mikkel/expipe/')
project = server.require_project('septum-mec')
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
        result.update({
            'name': name,
            'identifier': name
        })
        print('Uploading template', name)
        project.create_template(name=name, contents=result)
