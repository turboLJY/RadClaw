"""Token counting utilities."""


def estimate_tokens(text: str) -> int:
    return len(text) // 3