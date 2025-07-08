def format_model_name(model_name: str) -> str:
    return model_name.replace("/", "__").replace("-", "_").replace(".", "__").replace(":", "__").replace("@", "__")