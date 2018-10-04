import os.path as op
import os

def read_python(path):
    from six import exec_
    path = op.realpath(op.expanduser(path))
    assert op.exists(path)
    with open(path, 'r') as f:
        contents = f.read()
    metadata = {}
    exec_(contents, {}, metadata)
    metadata = {k.lower(): v for (k, v) in metadata.items()}
    return metadata

def write_python(path, contents, overwrite=False):
    assert isinstance(contents, dict)
    dir_path = op.dirname(path)
    if not op.exists(dir_path):
        os.mkdir(dir_path)
    assert op.isdir(dir_path)
    if not overwrite:
        assert not op.exists(path)
    def stringify(d):
        for k, v in d.items():
            return '{} = {}'.format(k, v)
    with open(path, 'w') as f:
        f.write(stringify(contents))
