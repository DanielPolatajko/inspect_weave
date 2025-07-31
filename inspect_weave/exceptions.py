class WeaveEvaluationException(Exception):
    """
    Exception raised when there is an error during the Inspect run.
    """
    def __init__(self, message: str, error: str):
        self.message = message
        self.error = error

    def __str__(self) -> str:
        return f"{self.message}: {self.error}"