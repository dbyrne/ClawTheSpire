"""Unified curriculum for BetaOne training.

Starter deck tiers: specific fights, each teaching one skill.
Random deck tiers: revisit hard fights with better cards.
Fixed-deck traps: force specific mechanics (shiv synergy, sly discard).

  T0:  Weak enemies              (100%)  — basics
  T1:  Normal enemies            (100%)  — block/attack/targeting
  T2:  Fogmog                    (100%)  — sustained damage
  T3:  Slugs + Spinner           (88%)   — multi-enemy management
  T4:  Cultists                  (90%)   — priority targeting
  T5:  Poison deck + Hard        (85%)   — poison archetype
  T6:  Shiv deck + Bosses        (95%)   — shiv archetype
  T7:  Pure shiv vs Byrdonis     (90%)   — Accuracy + Blade Dance trap
  T8:  Sly deck + Hard           (70%)   — sly archetype
  T9:  Sly discard vs Boss       (50%)   — Phrog (status) + Ceremonial Beast
  T10: Final exam                (avg)   — all tiers mixed
"""

from __future__ import annotations

import json
import random as stdlib_random
from dataclasses import dataclass, field
from pathlib import Path

from .deck_gen import build_random_deck_json, _make_starter, _card_defaults, lookup_card


# ---------------------------------------------------------------------------
# Fixed decks for specific tiers
# ---------------------------------------------------------------------------

# Shiv vs Strike deck: Strike is the trap, Blade Dance + Accuracy is the answer.
# 2 Strikes deal 12 total — not enough for 94 HP Byrdonis.
# 3 Blade Dances with Accuracy = 3 x 24 = 72 damage. Must play Accuracy first.
def _build_shiv_trap_deck() -> list[dict]:
    return ([lookup_card("DEFEND_SILENT") for _ in range(4)]
            + [lookup_card("STRIKE_SILENT") for _ in range(2)]
            + [lookup_card("BLADE_DANCE") for _ in range(3)]
            + [lookup_card("ACCURACY")])

# Sly discard deck: Tactician/Reflex cost 3 — awful to play, amazing to discard.
# Acrobatics and Survivor create discard choices where the model must pick Sly
# cards over Defend/Strike. Untouchable is playable Sly block (don't discard it
# unless you have a better Sly trigger like Tactician).
# 14 cards: enough to cycle through Acrobatics draws multiple times.
def _build_sly_discard_deck() -> list[dict]:
    return ([lookup_card("DEFEND_SILENT") for _ in range(2)]
            + [lookup_card("UNTOUCHABLE") for _ in range(2)]
            + [lookup_card("ACROBATICS") for _ in range(2)]
            + [lookup_card("SURVIVOR")]
            + [lookup_card("TACTICIAN") for _ in range(2)]
            + [lookup_card("REFLEX") for _ in range(2)]
            + [lookup_card("FLICK_FLACK") for _ in range(2)]
            + [lookup_card("STRIKE_SILENT")]
            + [lookup_card("NEUTRALIZE")])


# ---------------------------------------------------------------------------
# Tier configuration
# ---------------------------------------------------------------------------

@dataclass
class TierConfig:
    name: str
    deck_mode: str              # "starter" or "random"
    promote_threshold: float
    encounter_level: int = -1   # pool level (0-3), or -1 for custom
    custom_encounters: list[list[str]] | None = None
    custom_deck: list[dict] | None = None  # fixed deck override
    player_hp: int = 70
    deck_min_size: int = 12
    deck_max_size: int = 12
    deck_min_removals: int = 0
    deck_max_removals: int = 0
    deck_archetypes: list[str] | None = None  # restrict to these archetypes (None = all)


# Familiar hard encounters for random-deck tiers
_FAMILIAR_HARD = [
    ["CORPSE_SLUG", "CORPSE_SLUG", "SLUDGE_SPINNER"],
    ["CALCIFIED_CULTIST", "DAMP_CULTIST"],
    ["FOGMOG"],
]

# Previously impossible encounters — need random decks
_PREVIOUSLY_IMPOSSIBLE = [
    ["BYRDONIS"],
    ["BYGONE_EFFIGY"],
    ["TERROR_EEL"],
]


TIER_CONFIGS: list[TierConfig] = [
    # --- Starter deck: specific fights, one skill each ---
    TierConfig("Weak enemies",       deck_mode="starter", promote_threshold=1.00,
               encounter_level=0),
    TierConfig("Normal enemies",     deck_mode="starter", promote_threshold=1.00,
               encounter_level=1),
    TierConfig("Fogmog",             deck_mode="starter", promote_threshold=1.00,
               custom_encounters=[["FOGMOG"]]),
    TierConfig("Slugs + Spinner",    deck_mode="starter", promote_threshold=0.85,
               custom_encounters=[["CORPSE_SLUG", "CORPSE_SLUG", "SLUDGE_SPINNER"]]),
    TierConfig("Cultists",           deck_mode="starter", promote_threshold=0.90,
               custom_encounters=[["CALCIFIED_CULTIST", "DAMP_CULTIST"]]),

    # --- Random decks: learn one archetype at a time, then mix ---
    TierConfig("Poison deck + Hard",   deck_mode="random", promote_threshold=0.75,
               custom_encounters=_FAMILIAR_HARD,
               deck_archetypes=["poison"],
               deck_min_size=16, deck_max_size=20, deck_min_removals=1, deck_max_removals=3),
    TierConfig("Shiv deck + Bosses",    deck_mode="random", promote_threshold=0.95,
               custom_encounters=_PREVIOUSLY_IMPOSSIBLE,
               deck_archetypes=["shiv"],
               deck_min_size=16, deck_max_size=20, deck_min_removals=1, deck_max_removals=3),
    TierConfig("Pure shiv vs Byrdonis", deck_mode="custom", promote_threshold=0.55,
               custom_encounters=[["BYRDONIS"]],
               custom_deck=_build_shiv_trap_deck, player_hp=40),
    TierConfig("Sly deck + Hard",     deck_mode="random", promote_threshold=0.70,
               custom_encounters=_FAMILIAR_HARD + _PREVIOUSLY_IMPOSSIBLE,
               deck_archetypes=["sly"],
               deck_min_size=16, deck_max_size=20, deck_min_removals=1, deck_max_removals=3),
    TierConfig("Sly discard vs Boss", deck_mode="custom", promote_threshold=0.43,
               custom_encounters=[["CEREMONIAL_BEAST"], ["PHROG_PARASITE"]],
               custom_deck=_build_sly_discard_deck, player_hp=70),
]

# Auto-generate final "exam" tier: average threshold of all previous tiers
_avg_threshold = sum(c.promote_threshold for c in TIER_CONFIGS) / len(TIER_CONFIGS)
TIER_CONFIGS.append(
    TierConfig("Final exam", deck_mode="review_all",
               promote_threshold=round(_avg_threshold, 2))
)


# ---------------------------------------------------------------------------
# Default encounter pools (for T0/T1 when no encounter_pool.json)
# ---------------------------------------------------------------------------

DEFAULT_ENCOUNTER_POOLS: list[list[list[str]]] = [
    [["NIBBIT"], ["SHRINKER_BEETLE"], ["FUZZY_WURM_CRAWLER"]],
    [["LEAF_SLIME_S", "TWIG_SLIME_M"], ["CORPSE_SLUG", "CORPSE_SLUG"],
     ["INKLET", "INKLET", "INKLET"]],
    [],  # level 2 unused by starter tiers
    [],  # level 3 unused by starter tiers
]


# ---------------------------------------------------------------------------
# Curriculum
# ---------------------------------------------------------------------------

class CombatCurriculum:
    def __init__(
        self,
        encounter_pool_path: str | Path | None = None,
    ):
        self.tier = 0
        self.consecutive_good = 0
        self.gens_at_tier = 0
        self.deck_rng = stdlib_random.Random(42)

        if encounter_pool_path and Path(encounter_pool_path).exists():
            self.encounter_pools = self._load_from_pool(encounter_pool_path)
        else:
            self.encounter_pools = [list(p) for p in DEFAULT_ENCOUNTER_POOLS]

    def _load_from_pool(self, path: str | Path) -> list[list[list[str]]]:
        with open(path) as f:
            pool = json.load(f)

        # Load monster HP for validation
        monsters_path = (
            Path(__file__).resolve().parents[4]
            / "STS2-Agent" / "mcp_server" / "data" / "eng" / "monsters.json"
        )
        monster_hp: dict[str, int] = {}
        if monsters_path.exists():
            with open(monsters_path, encoding="utf-8") as f:
                for m in json.load(f):
                    monster_hp[m["id"]] = m.get("max_hp") or m.get("min_hp") or 20

        # Enemies with dedicated tiers — exclude from general pools
        reserved_enemies = set()
        for cfg in TIER_CONFIGS:
            if cfg.custom_encounters:
                for enc in cfg.custom_encounters:
                    reserved_enemies.update(enc)

        levels: list[list[list[str]]] = [[] for _ in range(4)]
        entries = pool if isinstance(pool, list) else list(pool.values())
        skipped = 0
        hp_caps = [60, 80, 90, 120]

        for enc in entries:
            if not isinstance(enc, dict):
                continue
            floor = enc.get("floor", 0)
            enemy_ids = enc.get("enemies", [])
            if not enemy_ids:
                continue
            if monster_hp:
                if any(eid not in monster_hp for eid in enemy_ids):
                    skipped += 1
                    continue
            if reserved_enemies and any(eid in reserved_enemies for eid in enemy_ids):
                continue
            total_hp = sum(monster_hp.get(eid, 0) for eid in enemy_ids) if monster_hp else 0

            if floor <= 4 and total_hp <= hp_caps[0]:
                levels[0].append(enemy_ids)
            elif floor <= 7 and total_hp <= hp_caps[1]:
                levels[1].append(enemy_ids)
            elif floor <= 12 and total_hp <= hp_caps[2]:
                levels[2].append(enemy_ids)
            elif floor <= 16 and total_hp <= hp_caps[3]:
                levels[3].append(enemy_ids)

        for i, level in enumerate(levels):
            if not level and i < len(DEFAULT_ENCOUNTER_POOLS) and DEFAULT_ENCOUNTER_POOLS[i]:
                levels[i] = list(DEFAULT_ENCOUNTER_POOLS[i])

        if skipped:
            print(f"Curriculum: skipped {skipped} encounters with missing monster data")

        return levels

    @property
    def config(self) -> TierConfig:
        return TIER_CONFIGS[min(self.tier, len(TIER_CONFIGS) - 1)]

    @property
    def max_tier(self) -> int:
        return len(TIER_CONFIGS) - 1

    @property
    def promote_threshold(self) -> float:
        return self.config.promote_threshold

    def _random_previous_tier(self) -> TierConfig:
        """Pick a random tier from T0 to current-1."""
        idx = stdlib_random.randint(0, self.tier - 1)
        return TIER_CONFIGS[idx]

    def sample_encounters(self, n: int) -> list[list[str]]:
        cfg = self.config

        if cfg.deck_mode == "review_all":
            encounters = []
            for _ in range(n):
                prev = self._random_previous_tier()
                if prev.custom_encounters:
                    encounters.append(stdlib_random.choice(prev.custom_encounters))
                elif prev.encounter_level >= 0:
                    encounters.append(stdlib_random.choice(self.encounter_pools[prev.encounter_level]))
                else:
                    encounters.append(stdlib_random.choice(self.encounter_pools[0]))
            return encounters

        if cfg.custom_encounters:
            return [stdlib_random.choice(cfg.custom_encounters) for _ in range(n)]

        pool = self.encounter_pools[cfg.encounter_level]
        review_level = max(0, cfg.encounter_level - 1)
        review_pool = self.encounter_pools[review_level]

        encounters = []
        for _ in range(n):
            if cfg.encounter_level > 0 and stdlib_random.random() < 0.2:
                encounters.append(stdlib_random.choice(review_pool))
            else:
                encounters.append(stdlib_random.choice(pool))
        return encounters

    def sample_deck_json(self) -> str:
        cfg = self.config

        if cfg.deck_mode == "review_all":
            prev = self._random_previous_tier()
            cfg = prev

        if cfg.custom_deck is not None:
            deck = cfg.custom_deck() if callable(cfg.custom_deck) else cfg.custom_deck
            return json.dumps(deck)
        elif cfg.deck_mode == "starter":
            return json.dumps(_make_starter())
        else:
            return build_random_deck_json(
                rng=self.deck_rng,
                min_size=cfg.deck_min_size,
                max_size=cfg.deck_max_size,
                min_removals=cfg.deck_min_removals,
                max_removals=cfg.deck_max_removals,
                archetypes=cfg.deck_archetypes,
            )

    def update(self, win_rate: float) -> str:
        self.gens_at_tier += 1

        if win_rate >= self.promote_threshold:
            self.consecutive_good += 1
            if self.consecutive_good >= 3 and self.tier < self.max_tier:
                self.tier += 1
                self.consecutive_good = 0
                self.gens_at_tier = 0
                return "promoted"
        else:
            self.consecutive_good = 0
        return "hold"

    def status_str(self) -> str:
        return f"T{self.tier} {self.config.name}"
