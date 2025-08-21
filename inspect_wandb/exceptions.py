class WeaveEvaluationException(Exception):
    """
    Exception raised when there is an error during the Inspect run.
    """
    def __init__(self, message: str, error: str):
        self.message = message
        self.error = error

    def __str__(self) -> str:
        return f"{self.message}: {self.error}"

class WandBNotInitialisedException(Exception):
    """
    Exception raised when W&B is not initialised.
    """
    def __init__(self):
        self.message = "wandb settings file not found. Please run `wandb init` to set up a project."

    def __str__(self) -> str:
        return self.message

class InvalidSettingsError(Exception):
    """
    Exception raised when the settings are invalid.
    """
    def __init__(self):
        self.message = "Settings must contain only 'weave' and 'models' keys"

    def __str__(self) -> str:
        return self.message
