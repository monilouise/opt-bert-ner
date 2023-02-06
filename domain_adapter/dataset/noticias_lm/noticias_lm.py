import datasets
import os
import sys
import json
from domain_adapter.dataset.config import PlainTextConfig

_DESCRIPTION = """\
See https://huggingface.co/datasets/harem.
"""

logger = datasets.logging.get_logger(__name__)

os.environ['HF_DATASETS_OFFLINE'] = '1'

class Noticias(datasets.GeneratorBasedBuilder):
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
                                    gen_kwargs={
                                        "filepath": os.path.join(sys.argv[1], 'treino_v1.jsonl')}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={
                                        "filepath": os.path.join(sys.argv[1], 'validacao_v1.jsonl')}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": os.path.join(sys.argv[1], 'teste_v1.jsonl')}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        data = [json.loads(line) for line in open(filepath, 'r', encoding='utf-8')]

        for row in data:
            id = row['id']
            text = row['text']
            yield id, {"text": text}

