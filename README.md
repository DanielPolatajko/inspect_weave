# `inspect_weave`
Integration with [Inspect](https://inspect.aisi.org.uk/) and [W&amp;B Weave](https://weave-docs.wandb.ai/)

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

### Weave setup

In order to utilise the Weave integration, you will also need to setup a Weave project, authenticate your environment with W&B, and set some relevant environment variables.

To get set up with a new Weave project. follow the instructions [here](https://weave-docs.wandb.ai/).

If you have an existing project, for example `test-project`, you should run 

```bash
wandb login
```

in the Inspect execution environment and follow the outlined steps.

You should then set the `WEAVE_PROJECT_NAME` environment variable with your project name (you can optionally prefix this with a Weave team name if you are part of multiple teams) e.g.

```bash
export WEAVE_PROJECT_NAME="test-project"
```

or with a team name, say `test-team`:

```bash
export WEAVE_PROJECT_NAME="test-team/test-project"
```

### Running Inspect with the integration

Once you have performed the above steps, the integration will be enabled for future Inspect runs in your environment by default. The Inspect logger output will link to the Weave dashboard where you can track and visualise eval results.


## Development

If you want to develop this project, you can fork and clone the repo and then run:

```bash
pip install -e ".[dev]"
pre-commit install
```

to install for development locally.

### Testing

We write unit tests with `pytest`. Currently these are very limited, but if you want to run the tests, you can simply run `pytest`.

## Project notes

This project in a work-in-progress, being developed as a [MARS](https://www.cambridgeaisafety.org/mars) project by [DanielPolatajko](https://github.com/DanielPolatajko), [Qi Guo](https://github.com/Esther-Guo), [Matan Shtepel](https://github.com/GnarlyMshtep), and supervised by Justin Olive. We are open to feature requests and suggestions for future directions (including extensions of this integration as well as other possible Inspect integrations).

### Compatability warning

This project currently depends on unreleased changes from the core Inspect library, so for the meantime we have the main branch of the Inspect repo in the project dependencies. This might cause some compatability issues, but we will update upon the next release.
