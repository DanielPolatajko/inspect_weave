import logging
import os
from pathlib import Path
from typing import Any
from typing_extensions import override

import pandas as pd
import wandb
from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, TaskStart
from inspect_ai.log import EvalSample
from inspect_ai.scorer import CORRECT
from inspect_viz import Component
from inspect_viz.plot import write_png_async

from inspect_viz.view.beta import scores_heatmap
from inspect_viz import Data
from inspect_ai.analysis.beta import evals_df
from inspect_weave.utils import parse_inspect_weave_settings, read_wandb_entity_and_project_name_from_settings, wandb_dir

logger = logging.getLogger(__name__)

class Metric:
    ACCURACY: str = "accuracy"
    SAMPLES: str = "samples"

class WandBModelHooks(Hooks):
    def __init__(self) -> None:
        self._correct_samples: int = 0
        self._total_samples: int = 0
        self.settings: dict[str, Any] | None = None

    @override
    def enabled(self) -> bool:
        self.settings = self.settings or parse_inspect_weave_settings()
        return self.settings["models"]["enabled"]

    @override
    async def on_run_start(self, data: RunStart) -> None:
        config_path = Path(wandb_dir()) / "inspect-weave-settings.yaml"
        entity, project_name = read_wandb_entity_and_project_name_from_settings(logger=logger)
        self.run = wandb.init(id=data.run_id, entity=entity, project=project_name) 

        wandb.save(config_path, base_path=config_path.parent, policy="now")
        assert self.settings is not None
        if self.settings["models"].get("config"):
            wandb.config.update(self.settings["models"]["config"])

        _ = self.run.define_metric(step_metric=Metric.SAMPLES, name=Metric.ACCURACY)

    @override
    async def on_run_end(self, data: RunEnd) -> None:
        try:
            logs = [log.location for log in data.logs]
            df = evals_df(logs)
            await self._log_scores_heatmap(data, df)
            self._log_summary(data)
        except Exception as e:
            logger.warning(f"Error in wandb_hooks: {e}")

        wandb.finish()

    @override
    async def on_task_start(self, data: TaskStart) -> None:
        inspect_tags = (
            f"inspect_task:{data.spec.task}",
            f"inspect_model:{data.spec.model}",
            f"inspect_dataset:{data.spec.dataset.name}",
        )
        if self.run.tags:
            self.run.tags = self.run.tags + inspect_tags
        else:
            self.run.tags = inspect_tags

    @override
    async def on_sample_end(self, data: SampleEnd) -> None:
        self._total_samples += 1
        if data.sample.scores:
            self._correct_samples += int(self._is_correct(data.sample))
            wandb.log(
                {Metric.SAMPLES: self._total_samples, Metric.ACCURACY: self._accuracy()}
            )

    async def _log_scores_heatmap(self, data: RunEnd, df: pd.DataFrame) -> None:
        viz_data = Data.from_dataframe(df)
        plot = scores_heatmap(viz_data, x="task_display_name", y="model", fill="score_headline_value")

        await self._log_image(data.run_id, plot, "scores_heatmap")

    async def _log_image(self, run_id: str, plot: Component, name: str) -> None:
        path = f"./.plots/{run_id}/{name}.png"
        if not os.path.exists(f"./.plots/{run_id}"):
            os.makedirs(f"./.plots/{run_id}")
        await write_png_async(path, plot)
        wandb.log({name: wandb.Image(path)})

    def _log_summary(self, data: RunEnd) -> None:
        summary = {
            "samples_total": self._total_samples,
            "samples_correct": self._correct_samples,
            "accuracy": self._accuracy(),
            "logs": [log.location for log in data.logs],
        }
        wandb.summary.update(summary)
        logger.info(f"WandB Summary: {summary}")

    def _is_correct(self, sample: EvalSample) -> bool:
        if not sample.scores:
            return False

        values = [score.value for score in sample.scores.values()]
        return CORRECT in values or 1 in values or 1.0 in values or True in values

    def _accuracy(self) -> float:
        if self._total_samples == 0:
            return 0.0

        return self._correct_samples * 1.0 / self._total_samples