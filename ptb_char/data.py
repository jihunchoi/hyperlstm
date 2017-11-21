from torchtext import data, datasets


class PTBChar(datasets.LanguageModelingDataset):

    def __init__(self, path, text_field):
        super().__init__(path=path, text_field=text_field)

    @classmethod
    def splits(cls, path='data/ptb_char',
               train='ptb.char.train.txt', validation='ptb.char.valid.txt',
               test='ptb.char.test.txt', **kwargs):
        assert path is not None, 'path parameter cannot be None'
        return super().splits(
            path=path, train=train, validation=validation,
            test=test, text_field=kwargs['text_field'])


class PTBCharTextField(data.Field):

    def __init__(self):
        super().__init__()
