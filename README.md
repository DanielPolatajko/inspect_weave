# `inspect_weave`
Integration with [Inspect](https://inspect.aisi.org.uk/) and Weights & Biases. Initially, this integration was focused primarily on [Weave](https://weave-docs.wandb.ai/), but we are also expanding to include the [wandb Models API](https://docs.wandb.ai/guides/models/)

## Demo Video

<div>
    <a href="https://www.loom.com/share/1578ad78581146d08348cfe2a13270b0">
      <p>WIP: Integrating Inspect Weave with Inspect AI for LLM Evaluations 🚀 - Watch Video</p>
    </a>
    <a href="https://www.loom.com/share/1578ad78581146d08348cfe2a13270b0">
      <img style="max-width:300px;" src="https://cdn.loom.com/sessions/thumbnails/1578ad78581146d08348cfe2a13270b0-d6183465b48a6d2b-full-play.gif">
    </a>
  </div>

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

#### Console configuration

If you have an existing project, for example `test-project`, you can run 

```bash
wandb login
```

in the directory where you will run Inspect from and follow the outlined steps.

You should then set the project which you'd like to write eval results to. This can be done with:

```bash
wandb init
```

`inspect_weave` will then by default use whichever project you set as default during the `wandb init` flow when writing to Weave.

#### Environment variables

If you are running Inspect in an automated environment where stepping through `wandb` CLI configurations is impractical, you can instead configure the integration with environment variables. To achieve an equivalent setup to the above, you can set the following env variables:

- `WANDB_API_KEY` - set this to your `wandb` API key for authentication
- `WANDB_ENTITY` - set this to the name of your `wandb` entity (i.e. team name)
- `WANDB_PROJECT` - set this to the `wandb` project which you would like to write data to

Environment variables take precedence over `wandb` settings set via the CLI, so if you want to override the settings, using env vars is a viable option.

There are also a handful of `wandb` environment variables which are not directly parsed by the inspect_weave integration, but will influence the behaviour of `wandb` if passed at runtime. These can be found [here](https://docs.wandb.ai/guides/track/environment-variables/)

### Configuration

`inspect_weave` works out-of-the-box after running `wandb init` - no additional configuration is required! By default, both Weave and Models integrations are enabled, using the project and entity from your wandb settings or set via env variables.

#### Optional Customization

For advanced users who want to customize the behavior, you can add a `[tool.inspect-weave]` section to your project's `pyproject.toml` file:

```toml
[tool.inspect-weave.weave]
enabled = true  # Enable/disable Weave integration (default: true)

[tool.inspect-weave.models]
enabled = false  # Enable/disable Models integration (default: true)
files = ["pyproject.toml", "log/*"]  # Files/folders to upload with Models run, path relative to your current working directory (default: none)
```

You can also manually set the `wandb` entity and project in `pyproject.toml` e.g.

```toml
[tool.inspect-weave.weave]
wandb_entity = "test-entity"
wandb_project = "test-project"

[tool.inspect-weave.models]
enabled = false  # Enable/disable Models integration (default: true)
wandb_entity = "test-entity"
wandb_project = "test-project"
files = ["pyproject.toml", "log/*"]  # Files/folders to upload with Models run, path relative to your current working directory (default: none)
```

#### Autopatching

For the Weave integration, there is an experimental autopatching feature which is disabled by default. This patches some Inspect functions with Weave tracing calls, such that the Weave traces UI displays a call trace which more closely resembles the structure of an Inspect eval (e.g. one call per sample, with child calls for each solver and scorer).

This feature can be configured with the `autopatch` parameter e.g.

```toml
[tool.inspect-weave.weave]
autopatch = true
```

or by setting the environment variable `INSPECT_WEAVE_WEAVE_AUTOPATCH=true`.

### Running Inspect with the integration

Once you have performed the above steps, the integration will be enabled for future Inspect runs in your environment by default. The Inspect logger output will link to the Weave dashboard where you can track and visualise eval results.

### Disabling the integration

You can disable either integration by adding configuration to your `pyproject.toml`. For example:

```toml
[tool.inspect-weave.weave]
enabled = false  # Disable Weave integration

[tool.inspect-weave.models]  
enabled = true   # Keep Models integration enabled
```

This would disable the Weave integration while keeping the Models API integration enabled.

#### Environment Variables

You can also enable or disable the integrations by setting the following environment variables:

- `INSPECT_WEAVE_MODELS_ENABLED`
- `INSPECT_WEAVE_WEAVE_ENABLED`

If the former is set to anything truthy, the Models integration will be enabled, and if it is set to anything falsey, the integration will be disabled. If it is unset, the settings loader will fallback to the `wandb` settings and `pyproject.toml` to determine whether to enable the integration.

The latter env var controls the Weave integration in the same manner.


## Development

If you want to develop this project, you can fork and clone the repo and then run:

```bash
uv sync --group dev
pre-commit install
```

to install for development locally.

### Testing

We write unit tests with `pytest`. If you want to run the tests, you can simply run `pytest`. Please consider writing a test if adding a new feature, and make sure that tests are passing before submitting changes.

## Project notes

This project in a work-in-progress, being developed as a [MARS](https://www.cambridgeaisafety.org/mars) project by [DanielPolatajko](https://github.com/DanielPolatajko), [Qi Guo](https://github.com/Esther-Guo), [Matan Shtepel](https://github.com/GnarlyMshtep), and supervised by Justin Olive. We are open to feature requests and suggestions for future directions (including extensions of this integration as well as other possible Inspect integrations).
