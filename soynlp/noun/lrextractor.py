import os


installpath = os.path.abspath(os.path.dirname(__file__))


class LRNonuExtractor():
    def __init__(
        self,
        max_left_length=10,
        max_right_length=9,
        pos_features=None,
        neg_features=None,
        postprocessing=None,
        verbose=True,
        debug_dir=None
    ):

        self.max_left_length = max_left_length
        self.max_right_length = max_right_length
        self.verbose = verbose
        self.debug_dir = debug_dir

        self.pos, self.neg, self.common = prepare_r_features(pos_features, neg_features)
        self.postprocessing = prepare_postprocessing(postprocessing)


def prepare_r_features(pos_features=None, neg_features=None):
    """
    Check `pos_features` and `neg_features`
    If the argument is not defined, soynlp uses default R features

    Args:
        pos_features (collection of str)
        neg_features (collection of str)

    Returns:
        pos_features (set of str) : positive feature set excluding common features
        neg_features (set of str) : negative feature set excluding common features
        common_features (set of str) : feature appeared in both `pos_features` and `neg_features`
    """
    def load_features(path):
        with open(path, encoding='utf-8') as f:
            features = [line.strip() for line in f]
        features = {feature for feature in features if feature}
        return features

    default_feature_dir = f'{installpath}/pretrained_models/'

    if pos_features is None:
        pos_features = load_features(f'{default_feature_dir}/lrnounextractor.features.pos.v2')
    elif isinstance(pos_features, str) and (os.path.exists(pos_features)):
        pos_features = load_features(pos_features)

    if neg_features is None:
        neg_features = load_features(f'{default_feature_dir}/lrnounextractor.features.neg.v2')
    elif isinstance(neg_features, str) and (os.path.exists(neg_features)):
        neg_features = load_features(neg_features)

    if not isinstance(pos_features, set):
        pos_features = set(pos_features)
    if not isinstance(neg_features, set):
        neg_features = set(neg_features)

    common_features = pos_features.intersection(neg_features)
    pos_features = {feature for feature in pos_features if feature not in common_features}
    neg_features = {feature for feature in neg_features if feature not in common_features}
    return pos_features, neg_features, common_features


def prepare_postprocessing(postprocessing):
    # NotImplemented
    return postprocessing