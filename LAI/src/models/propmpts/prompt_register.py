"""Module with utils to register prompts."""
from collections import defaultdict

from data.text_preprocessing import Languages

registry = defaultdict(dict)

def register_prompt(model_name: str, language: Languages):
    def decorator(func):
        registry[model_name][language] = func()
        return func
    return decorator

def get_prompt_registry():
    return registry