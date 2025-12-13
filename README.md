# This is the repository for the final project of CSCI1470 at Brown University

<img src="DL Junqi Poster.jpg" style="width: 1200px; height: 800px;"/>

To run the project, either:

a. 
1. run 
```bash
uv sync
```
2. activate the environment
3. run
```bash
python -u scripts/train_from_config.py configs/[your_training_config_yaml_file]
```

b. 
1. submit a batch job through: 
```bash
sbatch slurm_scripts/[batch_script_name]
```