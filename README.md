# inspect_weave
Integration with Inspect and W&amp;B Weave

## Initial Weave hook integration

I've implemented a basic hook which logs evaluation results to Weave. This is demonstrated in hellaswag.py, which you can run with:

```
inspect eval hellaswag.py --model anthropic/claude-3-haiku-20240307 --limit 5
```

to get some basic results to show up on Weave. Our Weave project is [here](https://wandb.ai/danielpolatajko-mars/test-project/weave/evaluations?view=evaluations_default) - you should have an email to join but lmk if not. Hopefully the docs make it clear how to set up locally.


