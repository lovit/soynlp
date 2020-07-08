from ._noun_ver2 import LRNounExtractor_v2


def extract_noun(corpus_path, method='v2', min_noun_score=0.3,
    min_noun_frequency=1, min_eojeol_frequency=1, max_left_length=10,
    verbose=True, min_num_of_features=1, **kargs):

    if method == 'v2':
        # Fix construction
        noun_extractor = LRNounExtractor_v2()
    else:
        raise ValueError('Available method is one of ["v1", "v2"]')

    # do something
