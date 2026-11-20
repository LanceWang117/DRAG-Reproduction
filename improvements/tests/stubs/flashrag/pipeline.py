"""Just enough of ``flashrag.pipeline`` to import the frozen DRAG module."""


class BasicPipeline:
    """Record construction arguments; evaluation is a pass-through.

    ``DebateAugmentedRAG`` only calls ``super().__init__`` and, in ``run``,
    ``self.evaluate``. Tests build instances with ``object.__new__`` and never
    reach either, so the stub stays inert.
    """

    def __init__(self, config, prompt_template=None):
        self.config = config
        self.prompt_template = prompt_template

    def evaluate(self, dataset, do_eval=True):
        return dataset
