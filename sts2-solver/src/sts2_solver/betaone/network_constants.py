"""Lightweight BetaOne architecture constants.

This module intentionally has no torch dependency so slim distributed workers
can compute code fingerprints without importing the training network.
"""

# These must match betaone/encode.rs constants exactly.
MAX_HAND = 10
CARD_STATS_DIM = 28
RELIC_DIM = 27
HAND_AGG_DIM = 3
TURN_COUNTERS_DIM = 3
POTION_SLOTS = 3
POTION_FIELDS = 6
POTIONS_DIM = POTION_SLOTS * POTION_FIELDS
BASE_STATE_DIM = 156 + TURN_COUNTERS_DIM + POTIONS_DIM
STATE_DIM = BASE_STATE_DIM + MAX_HAND * CARD_STATS_DIM + MAX_HAND
ACTION_DIM = 35
MAX_ACTIONS = 30
ARCH_VERSION = 3
