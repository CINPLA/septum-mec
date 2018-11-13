from imports import *
import os.path as op
import os
import json

overwrite = True
server = expipe.load_file_system(root='/home/mikkel/expipe/')
base_dir = op.join(op.abspath(op.dirname(op.expanduser(__file__))), 'templates')
project = server.require_project('septum-mec')
for template, val in project.templates.items():
    result = val.contents
    identifier = result.get('identifier')
    if identifier is None:
        continue

    fname = op.join(base_dir, identifier + '.json')
    if op.exists(fname) and not overwrite:
        raise FileExistsError(
            'The filename "' + fname + '" exists, set ovewrite to true.')
    os.makedirs(op.dirname(fname), exist_ok=True)
    print('Saving template "' + template + '" to "' + fname + '"')
    with open(fname, 'w') as outfile:
        result = expipe.backends.firebase.convert_to_firebase(result)
        json.dump(result, outfile,
                  sort_keys=True, indent=4)
