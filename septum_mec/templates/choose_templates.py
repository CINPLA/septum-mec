import os
import os.path as op
import json
from shutil import copyfile

from config import mikkel_septum_entorhinal as conf
templates = [t for k, v in conf.TEMPLATES.items() for t in v]


for root, dirs, files in os.walk('templates'):
    for fname in files:
        if not fname.endswith('.json'):
            continue
        group = op.split(root)[1]
        name = group + '_' + op.splitext(fname)[0]
        if name.startswith('mikkel_'):
            new_name = name.replace('mikkel_', '')
        else:
            new_name = name
        if name in templates:
            copyfile(op.join(root, fname), 'keep/' + new_name + '.json')
