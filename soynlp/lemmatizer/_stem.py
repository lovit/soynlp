def extract_domain_stem(prediction_scores, lrgraph, known_stem_L,
    ignore_L=None, min_eomi_score=0.3, min_eomi_frequency=100,
    min_stem_score=0.3, min_stem_frequency=100, min_num_of_unique_firstchar=4,
    min_entropy_of_firstchar=0.5, min_stem_entropy=1.5):

    # TODO
    stems = {}
    pos_l = {}
    return stems, pos_l