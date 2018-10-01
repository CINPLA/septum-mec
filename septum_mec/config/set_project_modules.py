import expipe
import os
import os.path as op
import json


with open('septum-mec.json', 'r') as infile:
    result = json.load(infile)
project = expipe.require_project('septum-mec')
project.require_module(name='settings', contents=result)
