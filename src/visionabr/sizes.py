"""Image size budgets for Qwen2.5-VL.

Qwen2.5-VL turns every 28x28 pixel square of an image into one image token.
So "at most N image tokens" is the same as "at most N * 784 pixels".
vLLM takes that pixel cap per request as mm_processor_kwargs.max_pixels.
"""
PIXELS_PER_TOKEN = 28 * 28

SIZE_BUDGETS: dict[str, int | None] = {
    "full" : None,
    "~1024" : 1024,
    "~512" : 512,
    "~256" : 256
}


def max_pixels_for_budget(budget_tokens: int) -> int:
    """Return the max_pixels value for an image token budget.

    >>> max_pixels_for_budget(512)
    401408
    """
    if isinstance(budget_tokens, bool) or not isinstance(budget_tokens, int):
        raise TypeError(
            f"budget_tokens must be an int, got {type(budget_tokens).__name__}"
        )
    if budget_tokens <= 0:
        raise ValueError(f"budget_tokens must be a positive number. Received {budget_tokens}")
    return budget_tokens * PIXELS_PER_TOKEN


def max_pixels_for_size(name: str) -> int | None:
    """Return max_pixels for a size name like "~512", or None for "full".

    >>> max_pixels_for_size("~1024")
    802816
    >>> max_pixels_for_size("~512")
    401408
    >>> max_pixels_for_size("~256")
    200704
    >>> max_pixels_for_size("full") is None
    True
    """
    if name not in SIZE_BUDGETS:
        valid = ", ".join(SIZE_BUDGETS)
        raise ValueError(f"unknown size {name!r}, expected one of: {valid}")
    budget = SIZE_BUDGETS[name]
    if budget is None:
        return None
    return max_pixels_for_budget(budget)