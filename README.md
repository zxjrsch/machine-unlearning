<h1 align="center">
  Machine Unlearning
</h1>

<p align="center">
    <a href="https://github.com/zxjrsch/mimu" target="_blank"><img src="https://img.shields.io/badge/GitHub-Code-4A90E2?style=for-the-badge&logo=github&logoColor=white" alt="GitHub"></a>
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
