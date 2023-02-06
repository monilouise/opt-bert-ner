import datasets
import os
import sys

from domain_adapter.dataset.config import PlainTextConfig

_DESCRIPTION = "A dataset for long text documents."

logger = datasets.logging.get_logger(__name__)

os.environ['HF_DATASETS_OFFLINE'] = '1'


class TextDocument(datasets.GeneratorBasedBuilder):
    BUILDER_CONFIGS = [
        PlainTextConfig(
            name="plain_text",
            version=datasets.Version("1.0.0", ""),
            description="Plain text",
        ),
    ]

    def _info(self):
        return datasets.DatasetInfo(
            description=_DESCRIPTION,
            features=datasets.Features(
                {
                    "text": datasets.Value("string"),
                }
            ),
        )

    def _split_generators(self, dl_manager):
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={"filepath": os.path.join(sys.argv[1], 'train')}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={"filepath": os.path.join(sys.argv[1], 'dev')}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        for filename in os.listdir(filepath):
            f = os.path.join(filepath, filename)
            with open(f, 'r', encoding='utf-8') as file:
                id = f[f.rfind(os.path.sep) + 1:f.rfind('.txt')]
                text = file.read()
                yield id, {"text": text}
