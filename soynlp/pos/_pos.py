class POSExtractor:

    def train_extract(self, sents):
        self.train(self, sents)
        return self.extract()

    def train(self, sents):
        raise NotImplemented

    def extract(self):
        wordtags = {}

        # TODO: something
        return wordtags