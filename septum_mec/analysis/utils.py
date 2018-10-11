import os
import exdir
import json

def create_notebook(exdir_path, channel_group=0):
    exob = exdir.File(exdir_path)
    analysis_path = str(exob.require_group('analysis').directory)
    currdir = os.path.dirname(os.path.abspath(__file__))
    fname = os.path.join(currdir, 'template_notebook.ipynb')
    with open(fname, 'r') as infile:
        notebook = json.load(infile)
    notebook['cells'][0]['source'] = ['exdir_path = r"{}"\n'.format(exdir_path),
                                      'channel_group = {}'.format(channel_group)]
    fnameout = os.path.join(analysis_path, 'analysis_notebook.ipynb')
    print('Generating notebook "' + fnameout + '"')
    with open(fnameout, 'w') as outfile:
            json.dump(notebook, outfile,
                      sort_keys=True, indent=4)
    return fnameout
