"""Stand-ins that refuse to build real components.

The frozen modules import these names but the integration tests inject fakes,
so any call here means a test tried to construct a real generator or retriever.
Failing loudly keeps a stubbed run from looking like a real one.
"""


def _unavailable(name):
    raise RuntimeError(
        f'{name} is not available in the test stub; inject a fake instead'
    )


def get_generator(config):
    _unavailable('flashrag.utils.get_generator')


def get_retriever(config):
    _unavailable('flashrag.utils.get_retriever')
