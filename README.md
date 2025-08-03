# `inspect_weave`
Integration with [Inspect](https://inspect.aisi.org.uk/) and Weights & Biases. Initially, this integration was focused primarily on [Weave](https://weave-docs.wandb.ai/), but we are also expanding to include the [wandb Models API](https://docs.wandb.ai/guides/models/)

The integration is implemented as an Inspect [hook](https://inspect.aisi.org.uk/extensions.html#hooks).

## Usage

### Installation

To use this integration, you can install the package in the environment where you are running an Inspect eval with:

__pip__
```bash
pip install git+https://github.com/DanielPolatajko/inspect_weave.git
```

__uv__
```bash
uv pip install git+https://github.com/DanielPolatajko/inspect_weave.git
```

If you intend to use the W&B Models API integration, you also need to subsequently install `chromium` with:

```bash
playwright install-deps chromium
```

### W&B setup

In order to utilise the W&B integration, you will also need to setup a W&B project, authenticate your environment with W&B, and initialise the wandb client.

To get set up with a new Weave project. follow the instructions [here](https://weave-docs.wandb.ai/), or to set up a W&B project, look [here](https://docs.wandb.ai/quickstart/) (they are basically the same, but it might be useful to follow the guide of the feature you're more interested in)

If you have an existing project, for example `test-project`, you should run 

```bash
wandb login
```

in the directory where you will run Inspect from and follow the outlined steps.

You should then set the project which you'd like to write eval results to. This can be done with:

```bash
wandb init
```

`inspect_weave` will then by default use whichever project you set as default during the `wandb init` flow when writing to Weave.

### Configuration

`inspect_weave` allows you to specify finer-grained configuration by creating an `inspect-weave-settings.yaml` file and placing it in the `wandb` directory (where the settings file is after running `wandb init`). An example file can be found [here](./wandb/inspect-weave-settings.yaml). Available settings include enabling/disabling Weave or Models integrations, specifying the project and entity to write to for both Models and Weave, passing wandb config to the Models API, and specifying files to associated with the Models run.

As mentioned above, by default, the integration will use whichever W&B project and entity you specified when running wandb init, unless overridden in the `inspect-weave-settings.yaml`.

### Running Inspect with the integration

Once you have performed the above steps, the integration will be enabled for future Inspect runs in your environment by default. The Inspect logger output will link to the Weave dashboard where you can track and visualise eval results.

### Disabling the integration

You can disable the Weave integration by updating the `inspect-weave-settings.yaml` and specifiying `enabled: false` for the relevant API. For example:

```yaml
models:
    enabled: true

weave:
    enabled: false
```

would write data to the models API, but disable the Weave integration.


## Development

If you want to develop this project, you can fork and clone the repo and then run:

```bash
pip install -e ".[dev]"
pre-commit install
```

to install for development locally.

### Testing

We write unit tests with `pytest`. If you want to run the tests, you can simply run `pytest`. Please consider writing a test if adding a new feature, and make sure that tests are passing before submitting changes.

## Project notes

This project in a work-in-progress, being developed as a [MARS](https://www.cambridgeaisafety.org/mars) project by [DanielPolatajko](https://github.com/DanielPolatajko), [Qi Guo](https://github.com/Esther-Guo), [Matan Shtepel](https://github.com/GnarlyMshtep), and supervised by Justin Olive. We are open to feature requests and suggestions for future directions (including extensions of this integration as well as other possible Inspect integrations).
