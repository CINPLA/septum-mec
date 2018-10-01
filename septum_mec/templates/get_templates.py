import expipe
import os.path as op
import os
import json

overwrite = True

base_dir = op.join(op.abspath(op.dirname(op.expanduser(__file__))), 'templates')
project = expipe.require_project('septum-mec')
for template, val in project.templates.items():
    result = val.to_dict()
    identifier = result.get('identifier')
    if identifier is None:
        continue

    fname = op.join(base_dir, identifier + '.json')
    if op.exists(fname) and not overwrite:
        raise FileExistsError('The filename "' + fname +
                              '" exists, set ovewrite to true.')
    os.makedirs(op.dirname(fname), exist_ok=True)
    print('Saving template "' + template + '" to "' + fname + '"')
    with open(fname, 'w') as outfile:
        result = expipe.core.convert_to_firebase(result)
        json.dump(result, outfile,
                  sort_keys=True, indent=4)
