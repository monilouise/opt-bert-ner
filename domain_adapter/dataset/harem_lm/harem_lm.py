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

class Harem(datasets.GeneratorBasedBuilder):
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
        mode = sys.argv[2]
        return [
            datasets.SplitGenerator(name=datasets.Split.TRAIN,
                                    gen_kwargs={
                                        "filepath": os.path.join(sys.argv[1], 'FirstHAREM-' + mode + '-train.json')}),
            datasets.SplitGenerator(name=datasets.Split.VALIDATION,
                                    gen_kwargs={
                                        "filepath": os.path.join(sys.argv[1], 'FirstHAREM-' + mode + '-dev.json')}),
            datasets.SplitGenerator(name=datasets.Split.TEST,
                                    gen_kwargs={"filepath": os.path.join(sys.argv[1], 'MiniHAREM-' + mode + '.json')}),
        ]

    def _generate_examples(self, filepath):
        """This function returns the examples in the raw (text) form."""
        logger.info("generating examples from = %s", filepath)

        with open(filepath, 'r', encoding='utf-8') as file:
            data = json.loads(file.read())
            for row in data:
                id = row['doc_id']
                text = row['doc_text']
                yield id, {"text": text}
