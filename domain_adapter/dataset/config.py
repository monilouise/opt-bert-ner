import datasets


class PlainTextConfig(datasets.BuilderConfig):

    def __init__(self, **kwargs):
        super(PlainTextConfig, self).__init__(**kwargs)