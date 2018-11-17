import ipywidgets as widgets
from IPython.display import display
import expipe
import pathlib

expipe_path = pathlib.Path('/home/mikkel/expipe/charlotte_pnn_mec')
project = expipe.require_project(expipe_path)
actions = project.actions.keys()
atributes = []

w = widgets.SelectMultiple(
    options=actions,
    description='Actions',
    disabled=False
)
def show():
    display(w)
