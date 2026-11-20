"""Inert replacement for ``flashrag.prompt.PromptTemplate``."""


class PromptTemplate:
    """Accept the frozen modules' template arguments and store them.

    ``model/prompt.py`` builds templates at import time with keyword arguments.
    Tests replace the template before calling the pipeline, so formatting is
    never exercised here.
    """

    def __init__(self, *args, **kwargs):
        self.args = args
        self.kwargs = kwargs

    def get_string(self, **kwargs):
        raise RuntimeError(
            'PromptTemplate.get_string is not available in the test stub; '
            'inject a fake template instead'
        )
