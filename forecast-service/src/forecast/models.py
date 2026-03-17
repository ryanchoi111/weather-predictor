import structlog

log = structlog.get_logger()

_model_cache: dict = {}

AVAILABLE_MODELS = ["fcn"]


def load_model(name: str):
    """Lazy-load and cache an Earth2Studio prognostic model."""
    if name in _model_cache:
        return _model_cache[name]

    if name == "fcn":
        from earth2studio.models.px import FCN
        model = FCN.load_model(FCN.load_default_package())
    else:
        raise ValueError(f"Unknown model: {name}. Available: {AVAILABLE_MODELS}")

    _model_cache[name] = model
    log.info("model_loaded", name=name)
    return model
