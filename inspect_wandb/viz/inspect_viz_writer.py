from inspect_viz import Component
from inspect_viz.plot import write_png_async
import logging
from inspect_viz.view.beta import scores_heatmap
from inspect_viz import Data
import os
import pandas as pd
import wandb
from inspect_ai.hooks import RunEnd
from inspect_ai.analysis import evals_df

logger = logging.getLogger(__name__)

class InspectVizWriter:
    """
    Class for managing the generation and writing of visualisations with inspect_viz.
    These visualisations are saved as images to the wandb Models run, if the extra is enabled.
    """

    async def log_scores_heatmap(self, data: RunEnd, run: wandb.Run) -> None:
        try:
            logs = [log.location for log in data.logs]
            run.config["logs"] = logs
            df = evals_df(logs)
            await self._log_scores_heatmap(data, df)
        except Exception as e:
            logger.warning(f"Error creating scores heatmap: {e}")

    async def _log_scores_heatmap(self, data: RunEnd, df: pd.DataFrame) -> None:
        viz_data = Data.from_dataframe(df)
        plot = scores_heatmap(viz_data, task_name="task_display_name", model_name="model", score_value="score_headline_value")

        await self._log_image(data.run_id, plot, "scores_heatmap")

    async def _log_image(self, run_id: str, plot: Component, name: str) -> None:
        path = f"./.plots/{run_id}/{name}.png"
        if not os.path.exists(f"./.plots/{run_id}"):
            os.makedirs(f"./.plots/{run_id}")
        await write_png_async(path, plot)
        wandb.log({name: wandb.Image(path)})