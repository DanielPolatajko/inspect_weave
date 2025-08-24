# Welcome to Inspect WandB's documentation!
========================================

**Inspect WandB** is a Python library for integrating the [Inspect AI framework](https://inspect.aisi.org.uk/) with WandB's [Models](https://wandb.ai/site/models/) and [Weave](https://wandb.ai/site/weave/).
Inspect is a framework for developing and executing LLM evaluations developed by UK AI Security Institute.
WandB Models and WandB Weave are tools for logging, managing, and visualizing AI models, where WandB Models is more broad and focused on training runs while WandB Weave is specifically for LLM evaluations.

### Quickstart

For detailed installation instructions, see {doc}`installation`.
```bash
pip install "inspect_wandb @ git+https://github.com/DanielPolatajko/inspect_wandb.git"
```

and if not already done:
```bash
wandb login
wandb init
```

Run any Inspect eval with:
```bash
inspect eval YOUR_EVAL     
```

In the terminal you should see:
```bash
wandb: Syncing run KvLbgngnUcJTEa3HxQj5zs
wandb: ⭐️ View project at https://wandb.ai/YOUR_TEAM_NAME/YOUR_PROJECT_NAME
wandb: 🚀 View run at https://wandb.ai/YOUR_TEAM_NAME/YOUR_PROJECT_NAME/runs/UID
```

Clicking the second link will take you to the WandB Weave UI tab for the eval.

Please see {doc}`tutorial` for more details on how to navigate and use the WandB Weave UI!

(features)=
### Features
Inspect WandB boasts the following features:
* **Zero Code Changes:** Inspect WandB can be installed to any existing Inspect project and works out of the box without any code changes. Requires `"inspect_ai >= 0.3.118"` since Inspect WandB depends on the recent "inspect hooks" feature.
* **Filtering across Inspect eval runs:** A common pain point with Inspect is the lack of a visualization/UI-friendly way to search and process data across eval runs. WandB Weave's rich filtering options solve this problem.
* **Comparison across Inspect eval runs:** In addition to filtering, WandB Weave offers UI-interactive ways to compare data across eval runs and across different models on the same eval.
* **Shareability & Persistence:** While evals are often developed and assessed collaboratively, by default, Inspect stores all logs locally, making it difficult for teams to share and collaborate and easy for data to be lost. WandB Models and WandB Weave natively store all the data in the cloud in a way that is easy for the entire team to access. WandB Models and WandB Weave are completely free for academic and personal use.


### Credits
Inspect WandB is developed by Daniel Polatajko, Qi Guo, and Matan Shtepel with Justin Olive's mentorship as part of the Mentorship for Alignment Research Students (MARS) 3.0.
We are grateful for invaluable feedback from Alex Remedios (UK AISI) and Sami Jawhar (METR) which shaped this package. 

```{toctree}
index.md
installation.md
tutorial.md
configuration.md
contributing.md
```