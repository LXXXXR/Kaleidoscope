REGISTRY = {}

from .rnn_agent import RNNAgent
from .n_rnn_agent import (
    NRNNAgent,
    NRNNAgent_1R3,
    Kalei_type_NRNNAgent_1R3,
)
from .rnn_ppo_agent import RNNPPOAgent
from .conv_agent import ConvAgent
from .ff_agent import FFAgent
from .central_rnn_agent import CentralRNNAgent
from .mlp_agent import MLPAgent
from .atten_rnn_agent import ATTRNNAgent

REGISTRY["rnn"] = RNNAgent
REGISTRY["n_rnn"] = NRNNAgent
REGISTRY["n_rnn_1R3"] = NRNNAgent_1R3
REGISTRY["Kalei_type_n_rnn_1R3"] = Kalei_type_NRNNAgent_1R3
REGISTRY["rnn_ppo"] = RNNPPOAgent
REGISTRY["conv_agent"] = ConvAgent
REGISTRY["ff"] = FFAgent
REGISTRY["central_rnn"] = CentralRNNAgent
REGISTRY["mlp"] = MLPAgent
REGISTRY["att_rnn"] = ATTRNNAgent
