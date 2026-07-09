class PipelineStageException(Exception):
    """Base exception for all pipeline stage failures."""
    def __init__(self, message, *, task_id=None, chapter_id=None, stage=None):
        self.task_id = task_id
        self.chapter_id = chapter_id
        self.stage = stage or self.__class__.__name__
        super().__init__(message)


class DetectionException(PipelineStageException):
    def __init__(self, message="Detection failed", **kwargs):
        super().__init__(message, **kwargs)


class OCRException(PipelineStageException):
    def __init__(self, message="OCR extraction failed", **kwargs):
        super().__init__(message, **kwargs)


class TranslationException(PipelineStageException):
    def __init__(self, message="Translation failed", **kwargs):
        super().__init__(message, **kwargs)


class InpaintingException(PipelineStageException):
    def __init__(self, message="Inpainting failed", **kwargs):
        super().__init__(message, **kwargs)


class RenderingException(PipelineStageException):
    def __init__(self, message="Rendering failed", **kwargs):
        super().__init__(message, **kwargs)