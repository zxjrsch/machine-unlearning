<p align="center">
  <img src="./assets/Minimally_Invasive_Machine_Unlearning.png" alt="MIMU Logo" width="100%" height="100%" style="vertical-align: middle;"/>
</p>

<p align="center">
    <a href="https://github.com/layer6ai-labs/MIMU" target="_blank"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
  <a href="https://deepwiki.com/zxjrsch/mimu"><img src="https://img.shields.io/badge/Deep Wiki-Online-00B89E?style=for-the-badge&logo=internet-explorer&logoColor=white" alt="Deep Wiki"></a>
</p>

### Current Experiments 

The current pipeline allows us to compute the following 5 metrics for unlearning and model performance in our MIMU method plus 3 additional baselines (SFT randomization, random masking, default). 

1. Cross entropy loss of classifier under forget set (measures unlearning)
2. Cross entropy loss of classifier under retain set (measures utility degradation)
3. Probability of classifying forget class averaged over a batch (measures unlearning)
4. Score (percent correct classification) under forget set (measures unlearning)
5. Score under retain set (measures utility degradation)

> [!NOTE] 
> Raw metrics are generated as `json` files saved at `eval/metrics_and_plots/json/<model>_<dataset>_top-{K}_kappa_{kappa}/*.json`.

The main object of our experiments is the `Pipeline` class that abstracts away all the details:

```python
# main.py

config = PipelineConfig(...)
pipeline = Pipeline(config)
pipeline.run()
```

An underneath the hood view of pipeline execution is as follows 

```
┌─────────────────┐
│    Pipeline     │
└─────────┬───────┘
          │
          ▼
┌─────────────────────────────┐
│  run_vision_model_training  │◄──── SupportedVisionModels
└─────────┬───────────────────┘      Vision Dataset
          │                          
          │    ┌─────────────────────────────┐
          └───►│  VisionModelTrainer         │
               └─────────┬───────────────────┘
                         │
                         ▼
                    ┌─────────┐
                    │ use_ddp?│
                    └────┬────┘
                    Yes  │  No
                  ┌──────┴──────┐
                  ▼             ▼
            ┌──────────┐  ┌──────────┐
            │train_ddp │  │  train   │
            └─────┬────┘  └─────┬────┘
                  │             │
                  └─────┬───────┘
                        ▼
      ┌─────────────────────────────────┐
      │  run_gcn_graph_generation       │
      └─────────┬───────────────────────┘
                │
                │    ┌─────────────────┐
                └───►│ GraphGenerator  │
                     └─────────┬───────┘
                               │
                               ▼
                ┌─────────────────────────────┐
                │    run_gcn_training         │◄──── GCNPriorDistribution
                └─────────┬───────────────────┘      Graph Dataset
                          │
                          │    ┌─────────────────┐
                          └───►│ GCNTrainer      │
                               └─────────┬───────┘
                                         │
                                         ▼
                       ┌─────────────────────────────────┐
                       │ run_single_evaluation_round     │◄──── SFTModes
                       └─────────┬───────────────────────┘      Checkpoint Files
                                 │
                                 │    ┌─────────────────┐
                                 └───►│       Eval      │
                                      └─────────┬───────┘
                                                ▼
        ┌─────────────────────────────────────────────────────┐
        │                     eval                            │
        │  ┌─────────────────────────────────────────────┐    │
        │  │        topK_list × kappa_list               │    │
        │  └─────────────────┬───────────────────────────┘    │
        │                    │                                │
        │                    ▼                                │
        │          ┌─────────────────┐                        │
        │     ┌───►│ More combos?    │                        │
        │     │    └─────────┬───────┘                        │
        │     │         Yes  │  No                            │
        │     └──────────────┘  │                             │
        │                       ▼                             │
        └───────────────────────────────────────────────────┐ │
                                                            │ │
                                                            ▼ ▼
                                                  ┌─────────────────┐
                                                  │ Results:        │
                                                  │ List[Dict]      │
                                                  └─────────────────┘
```



### Project structure 

```
mimu/
├── configs/                # configs 
├── tests/                  # unit tests 
├── main.py                 # entry point
├── pipeline.py             # end to end training and evaluation pipeline
├── data.py                 # runs inference to construct graph data for training GCN
├── eval.py                 # evaluation pipeline
├── model.py                # contains GCN and hooked models
├── trainer.py              # manages training
├── reporter.py             # plotting 
├── utils_data.py           # data utils
├── imagenet_classes.py     # key value pairs
├── .env                    # huggingface token
```

### Artifacts 

```
mimu/
├── reports/        # plots
├── eval/           # metrics 
├── checkpoints/    # gcn and classifier
├── datasets/       # train/eval data for gcn and classsifier
├── observability/  # training stats
```

### Supported Models and Datasets

We current support 2 models x 7 datasets

```python
# model.py
class SupportedVisionModels(Enum):
    HookedMLPClassifier = HookedMLPClassifier
    HookedResnet = HookedResnet
```

When adding new models, the last two layers should be feedforward layers without bias. For any exception the graph generator need to be modified,
see for example `GraphGenerator.flatten_in_out_activation_single_layer()` in `data.py`.

```python
# utils_data.py
class SupportedDatasets(Enum):
    MNIST = "MNIST"
    CIFAR10 = "CIFAR10"
    CIFAR100 = "CIFAR100"
    SVHN = "SVHN"
    IMAGENET_SMALL = "IMAGENET_SMALL"
    PLANT_CLASSIFICATION = "PLANT_CLASSIFICATION"
    POKEMON_CLASSIFICATION = "POKEMON_CLASSIFICATION"
```

⚠️ Whereas the first four datasets are loaded from torch (and saved in `./datasets/`), the last three need to be downloaded from huggingface, unzipped and placed in the appropriate path (see `utils_data.py`) and loaded with `get_unlearning_dataset` from `utils_data.py`.


| HF Dataset | Default Path | Actions |
| -----------|--------------|---------|
| [ImageNet](https://huggingface.co/datasets/ILSVRC/imagenet-1k) | `~/Datasets/ImageNet-small` | download and unzip |
| [Plant Classification](https://huggingface.co/datasets/jameelkhalidawan/Plant_Detection_Classification) | `~/Datasets/Plant-classification` | N.A. |
| [Pokemon Classification](https://huggingface.co/datasets/fcakyon/pokemon-classification) | `~/Datasets/Plant-classification` | N.A. |

To unzip ImageNet-small you can run 
```bash 
for f in *.tar.gz; do d="${f%.tar.gz}"; mkdir "$d" && tar -xzf "$f" -C "$d"; done
```

### How to run 

```bash
uv run main.py

# run in background and save log and returns pid 
nohup uv run main.py > test-2.log  2>&1 &

# or with venv activated
nohup python main.py > test.log  2>&1 &
```

> [!TIP]
> Highly recommended: clear artifacts before new runs to avoid data clashes, see `clean.sh`. Example command 

```bash
clear && bash clean.sh && uv run main.py 
```

Save output to file 

```bash 
nohup bash -c "clear && bash clean.sh && uv run main.py" > output.log 2>&1 &
```

### Possible Errors and Quick Fixes 

0. Some unit tests depend on artifacts to be genereated, for example GCN graph generation depends on vision model to be trained.
Some unit tests may contain outdated file naming convention or hardcoded paths. Be sure to update as necessary before running the tests.

1. If you run into cpu process errors, try reducing default number of workers for data loaders in `utils_data.py`

2. If you run into batch size mis-matches, set `is_train=True` in your retain or forget dataloader, since
validation set may not have enough datapoints for your specified batch, as was seen in SFT.

3. If you run into path errors while leveraging distributed training, this might be due to the framework automatically 
changing directory to a `temp` dir. The current codebase handles by default, including by telling the framework to init 
at the current working directory as opposed to `temp`. But future developments may break this.

4. Graphing is done differently with more dataset / model combinations.

### Adding models 

Define your hooked vision model with gradient getting hooks and model loader, see example in `model.py`.

### Adding Datasets

Define new dataset in `utils_data.py` and specialized unlearning dataloding class.


## How To Contribute 
### Adding Baselines

Experiments can be added to `eval.py` and outcomes plotted using `reporter.py`. Use `main.py` to launch your experiments.

```python 
# eval.py

class EvalConfig:
    # add relevant configs for new baselines here and modify main.py if necessary

class Eval:

    # get mask for given forget digit
    mask = self.get_model_mask()

    # measure unlearning, use forget set dataloader 
    forget_loader = self.mnist_forget_set()

    # measure utility degradation, use retain set dataloader
    retain_loader = self.mnist_retain_set()

    # inference baseline model, see method definition for return metrics
    metrics_dict = self.inference(
        model = your_model, 
        data_loader = your_eval_data_loader, 
        is_forget_set = is_your_data_forget_set,
        description = your_experiment_name
    )

    def draw_visualization(self):
        # visualize your new experiment by adding to this method 

    # add your baseline to 
    def eval_unlearning(self):
    
    # add your baseline to 
    def eval_performance_degradation(self):

    # add your baseline to 
    def eval_mask_efficacy(self):

    # run experiment and return metrics 
    def eval(self):

```

To plot your metrics modify `reporter.py`. The entry point is `Reporter.plot`

```python 
# reporter.py

class ReporterConfig:
    # add any configs if needed


class Reporter:

    # Add a set of data retrieval methods like in 

    ############ SFT Unlearning Baseline ############


    # Add your plots by modifying each draw_* method
    def draw_score_curves_on_forget_set(self) -> None:

        # ...

        y_sft = self.get_sft_unlearning_forget_set_score()
        plt.plot(x, y_sft, label='mimu topK maksing')

        # your baseline plot


```



### Dev Tools

```bash
uv run pre-commit run --all-files
```

```bash
# vision model training tracking 
trackio show
```

```bash
 uv run pytest -s

 # run specific test
 uv run pytest tests/test_utils_data.py::test_mnist -s
```

