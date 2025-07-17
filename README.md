# MIMU 

### Project structure 

```
mimu/
├── configs/        # configs 
├── data.py         # runs inference to construct graph data for training GCN
├── eval.py         # evaluation pipeline
├── main.py         # entry point
├── model.py        # contains GCN and hooked models
├── trainer.py      # manages training
├── reporter.py     # plotting 
```

### Artifacts 
```
mimu/
├── eval/           # metrics, plots 
├── checkpoints/    # gcn and classifier
├── datasets/       # train/eval data for gcn and classsifier
```

### How to run 

```bash
uv run main.py
```