from .base import FrameSatError


class InferenceError(FrameSatError):
    """
    Raised when model inference fails.
    """

    def __init__(
        self,
        message: str,
    ) -> None:
        super().__init__(
            message,
            code="INFERENCE_ERROR",
        )
