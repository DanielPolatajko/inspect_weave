(installation)=
# Installation

To use this integration, you should install the package in the Python environment where you are running Inspect - Inspect will automatically detect the hooks and utilise them during eval runs. The `inspect_wandb` integration has 3 components:

- **Models**: This integrates Inspect with the W&B Models API to store eval run statistics and configuration files for reproducability
- **Weave**: This integrates Inspect with the W&B Weave API which can be used to track and analyse eval scores, transcripts and metadata
- **Viz**: An experimental integration with [inspect_viz](https://github.com/meridianlabs-ai/inspect_viz) which allows you to generate visualisations using inspect viz and save them as images to the Models API run

By default, this integration will only install and enable the Models component, but the Weave and Viz components are easy to add as extras. To install just Models:

**pip**
```bash
pip install git+https://github.com/DanielPolatajko/inspect_wandb.git
```

**uv**
```bash
uv pip install git+https://github.com/DanielPolatajko/inspect_wandb.git
```

To install Models and Weave
NOTE uv pip install "git+https://github.com/DanielPolatajko/inspect_wandb.git[weave]" worked for me and not whats below with 'inspect_wandb @'

**pip**
```bash
uv pip install "git+https://github.com/DanielPolatajko/inspect_wandb.git[weave]"```

**uv**
```bash
uv pip install inspect_wandb @ "git+https://github.com/DanielPolatajko/inspect_wandb.git[weave]"
```

And to install Models, Weave and Viz

**pip**
```bash
pip install inspect_wandb @ "git+https://github.com/DanielPolatajko/inspect_wandb.git[weave,viz]"
```

**uv**
```bash
uv pip install inspect_wandb @ "git+https://github.com/DanielPolatajko/inspect_wandb.git[weave,viz]"
```

If you intend to use the Viz integration, you also need to subsequently install `chromium` with:

```bash
playwright install-deps chromium
```

### Docs
TODO