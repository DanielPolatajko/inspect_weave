from __future__ import annotations
from types import MethodType
from typing import Annotated, Any, TypeVar, Union, cast
import logging
from pydantic import (
    Field,
    PrivateAttr,
)

import weave
from weave.flow.dataset import Dataset
from weave.flow.eval import Evaluation, default_evaluation_display_name
from weave.flow.model import Model
from weave.trace.context import call_context
from weave.trace.context.weave_client_context import require_weave_client
from weave.trace.op import Op
from weave.flow.eval_imperative import  _active_evaluation_loggers, current_output, EvaluationLogger, current_predict_call, current_summary, IMPERATIVE_EVAL_MARKER



T = TypeVar("T")
ID = str
ScoreType = Union[float, bool, dict]

logger = logging.getLogger(__name__)

class CustomEvaluationLogger(EvaluationLogger):
    """This class provides an imperative interface for logging evaluations.

    An evaluation is started automatically when the first prediction is logged
    using the `log_prediction` method, and finished when the `log_summary` method
    is called.

    Each time you log a prediction, you will get back a `ScoreLogger` object.
    You can use this object to log scores and metadata for that specific
    prediction. For more information, see the `ScoreLogger` class.

    Example:
        ```python
        ev = EvaluationLogger()
        pred = ev.log_prediction(inputs, output)
        pred.log_score(scorer_name, score)
        ev.log_summary(summary)
        ```
    """

    eval_attributes: Annotated[
        dict[str, str],
        Field(
            default_factory=dict,
            description="(Optional): A dictionary of attributes to add to the evaluation.",
        ),
    ]

    _is_finalized: bool = PrivateAttr(False)

    @property
    def attributes(self) -> dict[str, Any]:
        return IMPERATIVE_EVAL_MARKER | self.eval_attributes

    def model_post_init(self, __context: Any) -> None:
        """Initialize the pseudo evaluation with the dataset from the model."""
        # Register this instance in the global registry for atexit cleanup
        _active_evaluation_loggers.append(self)

        # At this point dataset has already been processed by the validator
        # and converted to a Dataset object
        self._pseudo_evaluation = Evaluation(
            dataset=cast(Dataset, self.dataset),
            scorers=[],
        )

        # The following section is a "hacky" way to create Model and Evaluation
        # objects that "look right" to our object saving system.

        # --- Setup the model object ---
        @weave.op(name="Model.predict", enable_code_capture=False)
        def predict(self: Model, inputs: dict) -> Any:
            # Get the output from the context variable
            return current_output.get()

        self.model.__dict__["predict"] = MethodType(predict, self.model)

        # --- Setup the evaluation object ---
        @weave.op(name="Evaluation.evaluate", enable_code_capture=False)
        def evaluate(self: Evaluation, model: Model) -> None: ...

        @weave.op(name="Evaluation.predict_and_score", enable_code_capture=False)
        def predict_and_score(self: Evaluation, model: Model, example: dict) -> dict:
            predict_method = cast(Op, model.get_infer_method())
            with weave.attributes(IMPERATIVE_EVAL_MARKER):
                output, predict_call = predict_method.call(model, example)
                current_predict_call.set(predict_call)

            # This data is just a placeholder to give a sense of the data shape.
            # The actual output is explicitly replaced in ScoreLogger.finish.
            return {
                "output": output,
                "scores": {},
                "model_latency": None,
            }

        @weave.op(name="Evaluation.summarize", enable_code_capture=False)
        def summarize(self: Evaluation) -> dict:
            return cast(dict, current_summary.get())

        self._pseudo_evaluation.__dict__.update(
            {
                "evaluate": MethodType(evaluate, self._pseudo_evaluation),
                "predict_and_score": MethodType(
                    predict_and_score, self._pseudo_evaluation
                ),
                "summarize": MethodType(summarize, self._pseudo_evaluation),
            }
        )

        # Create the evaluation call
        wc = require_weave_client()
        self._evaluate_call = wc.create_call(
            display_name=self.name or default_evaluation_display_name,
            op=self._pseudo_evaluation.evaluate,
            inputs={
                "self": self._pseudo_evaluation,
                "model": self.model,
            },
            attributes=self.attributes,
        )
        assert self._evaluate_call is not None
        call_context.push_call(self._evaluate_call)

    def _finalize_evaluation(self, output: Any = None, exception: BaseException | None = None) -> None:
        """Handles the final steps of the evaluation: cleaning up predictions and finishing the main call."""
        if self._is_finalized:
            return

        self._cleanup_predictions()

        assert (
            self._evaluate_call is not None
        ), "Evaluation call should exist for finalization"

        # Finish the evaluation call
        wc = require_weave_client()
        # Ensure the call is finished even if there was an error during summarize or elsewhere
        try:
            wc.finish_call(self._evaluate_call, output=output, exception=exception)
        except Exception:
            # Log error but continue cleanup
            logger.error(
                "Failed to finish evaluation call during finalization.", exc_info=True
            )

        # Pop the call regardless of finish success
        try:
            call_context.pop_call(self._evaluate_call.id)
        except Exception:
            # If popping fails (e.g., context already unwound), log and ignore
            logger.warning("Failed to pop evaluation call from context.", exc_info=True)

        self._is_finalized = True

    def finish(self, exception: BaseException | None = None) -> None:
        """Clean up the evaluation resources explicitly without logging a summary.

        Ensures all prediction calls and the main evaluation call are finalized.
        This is automatically called if the logger is used as a context manager.
        """
        if self._is_finalized:
            return

        # Finalize with None output, indicating closure without summary
        self._finalize_evaluation(output=None, exception=exception)

        # Remove from global registry since we've manually finalized
        if self in _active_evaluation_loggers:
            _active_evaluation_loggers.remove(self)