def learning_mode(value):
    if value is True:
        return "first-turn"
    if value is False or value is None:
        return False
    if value not in ("first-turn", "continuous"):
        raise ValueError("Learning mode must be first-turn or continuous")
    return value


def learning_context(messages, mode):
    mode = learning_mode(mode)
    if not mode or len(messages) < 2 or messages[-1]["role"] != "user":
        return None
    context = messages[:-1]
    if mode == "first-turn" and any(
        message["role"] not in {"system", "developer"} for message in context
    ):
        return None
    return context
