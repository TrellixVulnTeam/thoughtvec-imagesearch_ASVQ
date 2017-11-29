import numpy as np

from .SentenceEncoderUniSkip import *
from .SentenceEncoderBiSkip import *

class SentenceEvaluator():
    def __init__(self):
        self.model = None
        self.input_ = None

    def materialize(self):
        if self.model is None:
            encoder = encoder_manager.EncoderManager()
            uniskip = SentenceEncoderUniSkip()
            uniskip.materialize(encoder_mgr=encoder)
            biskip = SentenceEncoderBiSkip()
            biskip.materialize(encoder_mgr=uniskip.model)
            self.model = biskip.model

    def evaluate(self, sentence):
        thoughtvec = self.model.encode([sentence])
        return np.squeeze(thoughtvec, axis=0)

    def evaluate_batch(self, batch_of_sentences):
        if self.model is None:
            self.materialize()

        batch_of_thoughtvecs = self.model.encode(batch_of_sentences)
        return batch_of_thoughtvecs

if __name__=='__main__':

    s1 = 'A black dog is sitting on a bench, looking out on a barren farm field.'
    s2 = 'The face of an orange cat is looking straight ahead.'

    sentenceevaluator = SentenceEvaluator()

    s1vec = sentenceevaluator.evaluate(s1)
    print(s1vec)
    s2vec = sentenceevaluator.evaluate(s2)
    print(s2vec)

    s12vecs = sentenceevaluator.evaluate_batch([s1, s2])
    print(s12vecs)