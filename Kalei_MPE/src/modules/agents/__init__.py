REGISTRY = {}

from .rnn_agent import RNNAgent, RNNAgent_1R3, Kalei_RNNAgent_1R3
from .rnn_ns_agent import RNNNSAgent
from .rnn_feature_agent import RNNFeatureAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["rnn_ns"] = RNNNSAgent
REGISTRY["rnn_feat"] = RNNFeatureAgent
REGISTRY["rnn_1R3"] = RNNAgent_1R3
REGISTRY["Kalei_rnn_1R3"] = Kalei_RNNAgent_1R3
