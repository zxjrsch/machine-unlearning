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
├── reports/        # plots
├── eval/           # metrics 
├── checkpoints/    # gcn and classifier
├── datasets/       # train/eval data for gcn and classsifier
```

### How to run 

```bash
uv run main.py
```

Highly recommended: clear artifacts before new runs to avoid data clashes, see `clean.sh`. Example command 

```bash
clear && bash clean.sh && uv run main.py 
```

### Metrics 

1. Cross entropy loss of classifier under forget set (measures unlearning)
2. Cross entropy loss of classifier under retain set (measures utility degradation)
3. Probability of classifying forget class averaged over a batch (measures unlearning)
4. Score (percent correct classification) under forget set (measures unlearning)
5. Score under retain set (measures utility degradation)

Raw metrics are generated as `json` files saved at `eval/Metrics and Plots/metrics/` and visualizations are plotted at `reports/`.



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