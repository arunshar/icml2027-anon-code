import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


import pytest


@pytest.fixture(scope="session", autouse=True)
def _skip_when_the_tokenizer_files_are_not_in_the_cache():
    """A call that loads a tokenizer and finds neither the network nor the cache skips its test and says
    why. README.md tells how to fetch the small tokenizer files once."""
    try:
        import transformers
    except ImportError:
        yield
        return
    load = transformers.AutoTokenizer.from_pretrained

    def guarded(cls, *args, **kwargs):
        try:
            return load(*args, **kwargs)
        except OSError as error:
            pytest.skip("the tokenizer files of the pinned model are not in the Hugging Face cache and the "
                        "network is off (see README.md): " + str(error).splitlines()[0][:120])

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(transformers.AutoTokenizer, "from_pretrained", classmethod(guarded))
        yield
