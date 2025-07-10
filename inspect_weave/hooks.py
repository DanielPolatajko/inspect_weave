
import os

from inspect_ai.hooks import Hooks, RunEnd, RunStart, SampleEnd, hooks, TaskStart, TaskEnd
import weave
from weave.trace import weave_client
from inspect_weave.utils import format_model_name

@hooks(name="weave_evaluation_hooks", description="Integration hooks for writing evaluation results to Weave")
class WeaveEvaluationHooks(Hooks):
    """
    Provides Inspect hooks for writing eval scores to the Weave Evaluations API.
    """

    weave_eval_logger: weave.EvaluationLogger | None = None
    weave_trace_client: weave_client.WeaveClient | None = None


    async def on_run_start(self, data: RunStart) -> None:
        self.weave_trace_client = weave.init(os.environ["WEAVE_PROJECT_NAME"])    

    async def on_run_end(self, data: RunEnd) -> None:
        weave.finish()

    async def on_task_start(self, data: TaskStart) -> None:
        model_name = format_model_name(data.spec.model) 
        evaluation_name = f"{data.spec.task}_{data.spec.run_id}"
        self.weave_eval_logger = weave.EvaluationLogger(name=evaluation_name, dataset=data.spec.dataset.name or "test_dataset", model=model_name)

    async def on_task_end(self, data: TaskEnd) -> None:
        assert self.weave_eval_logger is not None
        self.weave_eval_logger.log_summary()
        self.weave_eval_logger.finish()

        from inspect_ai.analysis.beta import EvalResults
        from inspect_ai.analysis.beta._dataframe.record import import_record
        from inspect_ai.analysis.beta._dataframe.util import records_to_pandas
        # from inspect_ai.analysis.beta._dataframe.evals.table import reorder_evals_df_columns

        record = [import_record(data.log, data.log, EvalResults)]
        evals_table = records_to_pandas(record)
        # evals_table = reorder_evals_df_columns(evals_table, EvalResults)
        call = self.weave_trace_client.create_call(
            op="test_trace",
            inputs={
                "eval": evals_table
            }
        )
        self.weave_trace_client.finish_call(call=call)

    async def on_sample_end(self, data: SampleEnd) -> None:
        assert self.weave_eval_logger is not None
        sample_score_logger = self.weave_eval_logger.log_prediction(
            inputs={"input": data.sample.input},
            output=data.sample.output.completion
        )
        if data.sample.scores is not None:
            for k,v in data.sample.scores.items():
                sample_score_logger.log_score( # TODO: could we use the async method here?
                    scorer=k,
                    score=v.value if not isinstance(v.value, str) and not isinstance(v.value, list) else {"score": str(v.value)}  # TODO: handle different score return types
                 )
                if v.metadata is not None and "category" in v.metadata:
                    sample_score_logger.log_score(
                        scorer=f"{k}_{v.metadata['category']}",
                        score=v.value if not isinstance(v.value, str) and not isinstance(v.value, list) else {"score": str(v.value)}
                    )
            sample_score_logger.finish()

    def enabled(self) -> bool:
        weave_project_present = os.environ.get("WEAVE_PROJECT_NAME") is not None
        if not weave_project_present:
            raise ValueError("WEAVE_PROJECT_NAME is not set, must be set in the environment to use Weave Evaluation Hooks")
        return True
        # return False
    
@hooks(name="log_hooks", description="test log dataframes feature from the inspect_ai.analysis.beta module")
class LogHooks(Hooks):
    """
    Provides Inspect hooks for logging dataframes.
    """

    async def on_run_start(self, data: RunStart) -> None:
        pass

    async def on_run_end(self, data: RunEnd) -> None:
        pass

    async def on_task_start(self, data: TaskStart) -> None:
        pass

    async def on_task_end(self, data: TaskEnd) -> None:
        try:
            from inspect_ai.analysis.beta import evals_df, EvalResults
            from inspect_ai.analysis.beta._dataframe.record import import_record
            from inspect_ai.analysis.beta._dataframe.util import records_to_pandas
            from inspect_ai.analysis.beta._dataframe.evals.table import reorder_evals_df_columns
            record = [import_record(data.log, data.log, EvalResults)]
            evals_table = records_to_pandas(record)
            evals_table = reorder_evals_df_columns(evals_table, EvalResults)
            print(evals_table.columns)
        except Exception as e:
            import traceback
            print(f"An error occurred: {e}")
            print("Traceback:")
            traceback.print_exc()
    
    async def on_sample_end(self, data: SampleEnd) -> None:
        # TODO: test logging of samples: EvalSample, columns as SampleSummary(https://inspect.aisi.org.uk/reference/inspect_ai.analysis.html#samplesummary)
        pass


    def enabled(self) -> bool:
        return False