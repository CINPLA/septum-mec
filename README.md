# septum-mec
Analysis for the septum-mec project

pip install -r requirements.txt

export SEPTUM_MEC_DATA="/path/to/data"

### Educloud
SEPTUM_MEC_DATA="/projects/ec109/Mikkel/septum-mec/"

unload Ipython module in jupyter lab

in a terminal write

```bash
module purge
module load Miniconda3/22.11.1-1
conda activate /projects/ec109/conda-envs/septum-mec
cp -r /projects/ec109/conda-envs/ipykernels/septum-mec ~/.local/share/jupyter/kernels/
```

You might have to restart to see the septum-mec kernel in jupyter notebook
