"""
Configuration parameters for FedSC experiments.
"""

SEC_AGG_CONFIG = {
    "dropout_aggregation": True,
    "sec_agg_prev_clients": 3,
    "sec_agg_threshold": 0.7,
    "sec_agg_srcs": ["driver"],
}

DEFAULT_PARAMS = {
    "num_rounds": 5,
    "num_clients": 3,
    "n_clusters": 3,
    "consensus_alpha": 0.7,  # From Eq 5 in Sec 3.2
    "dp_epsilon": 1.0,       # Differential privacy budget
}
