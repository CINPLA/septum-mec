import expipe
import os
import os.path as op
import json

server = expipe.load_file_system(root='/home/mikkel/expipe/')
with open('septum-mec.json', 'r') as infile:
    result = json.load(infile)
project = server.require_project('septum-mec')
project.create_module(name='settings', contents=result)
