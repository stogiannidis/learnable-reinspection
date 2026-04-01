from .config import ReInspectionConfig

__all__ = [
    "InternVL3WithReInspection",
    "ReInspectionConfig",
    "load_model",
    "load_processor",
]


def __getattr__(name):
    if name == "InternVL3WithReInspection":
        from .modeling import InternVL3WithReInspection

        return InternVL3WithReInspection
    if name == "load_model":
        from .modeling import load_model

        return load_model
    if name == "load_processor":
        from .modeling import load_processor

        return load_processor
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
