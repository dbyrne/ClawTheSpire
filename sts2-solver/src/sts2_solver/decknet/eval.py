"""DeckNet eval harness: curated deck-composition decisions.

Each scenario sets up a specific (deck, run_state, candidate_mods) and
asserts that DeckNet's V-argmax matches the expected_best_idx.

Mirrors the BetaOne combat eval pattern: a small, curated, human-reviewed
set of "right answers" that we use to probe whether the model has learned
specific deck-level decisions independent of overall run WR.

Phase 0 scope: card reward picks (including skip). ~15 scenarios spanning:
- Obvious additions (take the synergy card)
- Obvious skips (don't bloat a lean deck with a filler)
- Archetype awareness (poison vs shiv vs draw)
- Deck size sensitivity (thin-deck premium for exhaust combos)
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field

import torch

from .encoder import encode_batch, encode_state
from .network import DeckNet
from .state import (
    CardRef, DeckBuildingState, DeckModification, ModKind, MapRoom,
    PlayerStats, RoomType, apply_mod,
)


# ---------------------------------------------------------------------------
# Deck builders for common contexts
# ---------------------------------------------------------------------------

def silent_starter() -> list[CardRef]:
    """Canonical Silent starter: 5 Strikes + 5 Defends + Neutralize + Survivor."""
    return (
        [CardRef("STRIKE_SILENT")] * 5
        + [CardRef("DEFEND_SILENT")] * 5
        + [CardRef("NEUTRALIZE"), CardRef("SURVIVOR")]
    )


def poison_deck(size: int = 20) -> list[CardRef]:
    """A poison-archetype deck: starter + several poison cards + cycling."""
    base = silent_starter()[:8]    # 4 strikes + 4 defends
    base += [
        CardRef("DEADLY_POISON"), CardRef("DEADLY_POISON"),
        CardRef("NOXIOUS_FUMES"), CardRef("BOUNCING_FLASK"),
        CardRef("POISONED_STAB"), CardRef("POISONED_STAB"),
        CardRef("CATALYST") if False else CardRef("ACCELERANT"),  # Catalyst isn't in v1 vocab
        CardRef("BANDIS"),  # placeholder; may not exist
    ]
    # Truncate known-missing ids by filtering to cards we've confirmed in vocab;
    # this list is a hint — the eval loader will silently drop unknowns.
    base = [c for c in base if c.id not in {"BANDIS", "CATALYST"}]
    while len(base) < size:
        base.append(CardRef("STRIKE_SILENT"))
    return base[:size]


def shiv_deck(size: int = 20) -> list[CardRef]:
    base = silent_starter()[:6]    # 3 strikes + 3 defends
    base += [
        CardRef("BLADE_DANCE"), CardRef("BLADE_DANCE"),
        CardRef("CLOAK_AND_DAGGER"), CardRef("ACCURACY"),
        CardRef("INFINITE_BLADES"), CardRef("STORM_OF_STEEL"),
        CardRef("DAGGER_THROW"), CardRef("DAGGER_THROW"),
    ]
    while len(base) < size:
        base.append(CardRef("STRIKE_SILENT"))
    return base[:size]


def thin_grand_finale_deck() -> list[CardRef]:
    """Deliberately thin deck built around Grand Finale (fires when draw empty)."""
    return [
        CardRef("DEFEND_SILENT"), CardRef("DEFEND_SILENT"),
        CardRef("STRIKE_SILENT"),
        CardRef("GRAND_FINALE"),
        CardRef("ACROBATICS"), CardRef("PREPARED"),
        CardRef("CALCULATED_GAMBLE"), CardRef("BURST"),
        CardRef("NEUTRALIZE"),
    ]


# ---------------------------------------------------------------------------
# Scenario dataclass
# ---------------------------------------------------------------------------

@dataclass
class DeckScenario:
    name: str
    description: str
    state: DeckBuildingState
    candidates: list[DeckModification]
    best_idx: int
    bad_idx: list[int] = field(default_factory=list)
    category: str = "uncategorized"
    # Alternative acceptable picks (beyond best_idx). A pick is "correct" if
    # it matches best_idx OR any index in acceptable_idx.
    acceptable_idx: list[int] = field(default_factory=list)


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def build_scenarios() -> list[DeckScenario]:
    out: list[DeckScenario] = []

    # --- Starter-deck card rewards ---

    out.append(DeckScenario(
        name="starter_take_blade_dance",
        category="starter_pick",
        description="Silent starter, offered Blade Dance — solid shiv-starter archetype pick",
        state=DeckBuildingState(
            deck=silent_starter(),
            player=PlayerStats(hp=65, max_hp=70, gold=50),
            relics=frozenset({"RING_OF_THE_SNAKE"}),
            act=1, floor=2,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("BLADE_DANCE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("SLIMED")),       # obviously bad
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),  # neutral
            DeckModification(kind=ModKind.IDENTITY),                             # skip
        ],
        best_idx=0,
        bad_idx=[1],
    ))

    out.append(DeckScenario(
        name="starter_skip_another_strike",
        category="bloat_skip",
        description="Starter already has 5 Strikes — adding another is bloat",
        state=DeckBuildingState(
            deck=silent_starter(),
            player=PlayerStats(hp=60, max_hp=70, gold=20),
            act=1, floor=3,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),  # skip is correct
        ],
        best_idx=1,
    ))

    # --- Archetype synergy ---

    out.append(DeckScenario(
        name="poison_deck_take_deadly",
        category="archetype_poison",
        description="Poison archetype with 2 existing poison cards — take another Deadly Poison",
        state=DeckBuildingState(
            deck=poison_deck(18),
            player=PlayerStats(hp=55, max_hp=70, gold=100),
            relics=frozenset({"RING_OF_THE_SNAKE"}),
            act=1, floor=8,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DEADLY_POISON")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="poison_deck_skip_blade_dance",
        category="archetype_poison",
        description="Pure poison deck — Blade Dance's shivs don't stack with poison synergy",
        state=DeckBuildingState(
            deck=poison_deck(19),
            player=PlayerStats(hp=50, max_hp=70, gold=150),
            act=2, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("BLADE_DANCE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEADLY_POISON")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
    ))

    out.append(DeckScenario(
        name="shiv_deck_take_infinite_blades",
        category="archetype_shiv",
        description="Shiv-heavy deck — Infinite Blades is an end-game engine card",
        state=DeckBuildingState(
            deck=shiv_deck(18),
            player=PlayerStats(hp=60, max_hp=70, gold=0),
            act=2, floor=3,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("INFINITE_BLADES")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="shiv_deck_take_accuracy",
        category="archetype_shiv",
        description="Shiv deck, no Accuracy yet — Accuracy scales all shiv damage",
        state=DeckBuildingState(
            deck=shiv_deck(17),
            player=PlayerStats(hp=55, max_hp=70, gold=50),
            act=2, floor=6,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("ACCURACY")),
            DeckModification(kind=ModKind.ADD, card=CardRef("SLIMED")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
        bad_idx=[1],
    ))

    # --- Deck size sensitivity ---

    out.append(DeckScenario(
        name="thin_grand_finale_skip_bloat",
        category="deck_size",
        description="Thin Grand Finale deck — adding non-combo cards breaks the cycle-empty condition",
        state=DeckBuildingState(
            deck=thin_grand_finale_deck(),
            player=PlayerStats(hp=55, max_hp=70, gold=40),
            act=2, floor=10,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),    # bloat
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),    # bloat
            DeckModification(kind=ModKind.ADD, card=CardRef("ACROBATICS")),       # supports cycle
            DeckModification(kind=ModKind.IDENTITY),                                # save the thin-ness
        ],
        best_idx=3,
        bad_idx=[0, 1],
    ))

    out.append(DeckScenario(
        name="thin_grand_finale_take_cycle",
        category="deck_size",
        description="Thin Grand Finale deck — Calculated Gamble is a perfect cycle tool",
        state=DeckBuildingState(
            deck=thin_grand_finale_deck(),
            player=PlayerStats(hp=60, max_hp=70, gold=0),
            act=1, floor=14,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("CALCULATED_GAMBLE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
        bad_idx=[1],
    ))

    # --- Bloat avoidance ---

    out.append(DeckScenario(
        name="skip_status_card",
        category="bloat_skip",
        description="Offered a curse-like card (Slimed) as reward — always skip",
        state=DeckBuildingState(
            deck=silent_starter(),
            player=PlayerStats(hp=70, max_hp=70, gold=0),
            act=1, floor=2,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("SLIMED")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
        bad_idx=[0],
    ))

    out.append(DeckScenario(
        name="large_deck_skip_filler",
        category="deck_size",
        description="30-card deck — extra Strikes just dilute draw quality",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("STRIKE_SILENT")] * 20,
            player=PlayerStats(hp=40, max_hp=70, gold=80),
            act=2, floor=8,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
        bad_idx=[0],
    ))

    # --- Key high-value cards ---

    out.append(DeckScenario(
        name="offered_footwork_always_take",
        category="utility",
        description="Footwork is a Power giving permanent Dex — almost always correct to take",
        state=DeckBuildingState(
            deck=silent_starter(),
            player=PlayerStats(hp=65, max_hp=70, gold=50),
            act=1, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("FOOTWORK")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="offered_prepared_in_early",
        category="utility",
        description="Prepared is a 0-cost cycle card — great early add",
        state=DeckBuildingState(
            deck=silent_starter(),
            player=PlayerStats(hp=70, max_hp=70, gold=30),
            act=1, floor=1,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("PREPARED")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    # --- Late-game HP pressure ---

    out.append(DeckScenario(
        name="low_hp_skip_risky_adds",
        category="late_game_hp",
        description="Low HP act 2 — Wraith Form costs 3 which is a lot when behind on curve",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("DEADLY_POISON"), CardRef("BLADE_DANCE")],
            player=PlayerStats(hp=20, max_hp=70, gold=20),
            act=2, floor=10,
            map_ahead=[MapRoom(RoomType.BOSS, 2)],
            boss_id="TIME_EATER",
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("WRAITH_FORM")),
            DeckModification(kind=ModKind.ADD, card=CardRef("FOOTWORK")),
            DeckModification(kind=ModKind.ADD, card=CardRef("SURVIVOR")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        # Footwork (permanent Dex) is safer add than Wraith Form at low HP + act 2
        best_idx=1,
    ))

    out.append(DeckScenario(
        name="act3_thin_deck_skip_bloat",
        category="deck_size",
        description="Act 3 lean deck — save it for the Heart run",
        state=DeckBuildingState(
            deck=thin_grand_finale_deck() + [CardRef("GRAND_FINALE"), CardRef("BURST")],
            player=PlayerStats(hp=45, max_hp=70, gold=200),
            act=3, floor=1,
            boss_id="CORRUPT_HEART",
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=2,
    ))

    # --- Sly synergy ---

    out.append(DeckScenario(
        name="sly_deck_take_tactician",
        category="archetype_cycle",
        description="Deck with Prepared + Reflex — Tactician chains Sly value with discards",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("PREPARED"), CardRef("REFLEX"), CardRef("ACROBATICS")],
            player=PlayerStats(hp=60, max_hp=70, gold=50),
            act=1, floor=7,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("TACTICIAN")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.extend(_build_expanded_scenarios())
    return out


# ---------------------------------------------------------------------------
# Additional deck builders for expanded scenarios
# ---------------------------------------------------------------------------

def heavy_strike_deck(size: int = 18) -> list[CardRef]:
    base = silent_starter() + [CardRef("STRIKE_SILENT")] * 3  # 8 strikes total
    while len(base) < size:
        base.append(CardRef("DEFEND_SILENT"))
    return base[:size]


def heavy_defend_deck(size: int = 18) -> list[CardRef]:
    base = silent_starter() + [CardRef("DEFEND_SILENT")] * 3  # 8 defends total
    while len(base) < size:
        base.append(CardRef("STRIKE_SILENT"))
    return base[:size]


def heavy_powers_deck(size: int = 20) -> list[CardRef]:
    """Deck with 5 Powers — risk of power-overload."""
    base = silent_starter()[:8]
    base += [
        CardRef("FOOTWORK"), CardRef("NOXIOUS_FUMES"),
        CardRef("ACCURACY"), CardRef("WELL_LAID_PLANS"),
        CardRef("INFINITE_BLADES"),
    ]
    while len(base) < size:
        base.append(CardRef("STRIKE_SILENT"))
    return base[:size]


def block_focused_deck(size: int = 18) -> list[CardRef]:
    """Silent deck leaning into durability."""
    base = silent_starter()[:10]
    base += [CardRef("FOOTWORK"), CardRef("BLUR"), CardRef("DEFEND_SILENT")]
    while len(base) < size:
        base.append(CardRef("DEFEND_SILENT"))
    return base[:size]


def cycle_deck(size: int = 18) -> list[CardRef]:
    """Draw/cycle archetype."""
    base = silent_starter()[:8]
    base += [
        CardRef("PREPARED"), CardRef("PREPARED"),
        CardRef("ACROBATICS"), CardRef("ESCAPE_PLAN"),
        CardRef("CALCULATED_GAMBLE"),
    ]
    while len(base) < size:
        base.append(CardRef("STRIKE_SILENT"))
    return base[:size]


def exhaust_thin_deck() -> list[CardRef]:
    """Thin deck with exhaust payoffs (like Grand Finale)."""
    return [
        CardRef("DEFEND_SILENT"), CardRef("DEFEND_SILENT"),
        CardRef("STRIKE_SILENT"),
        CardRef("OMNISLICE"), CardRef("GRAND_FINALE"),
        CardRef("ACROBATICS"), CardRef("BURST"),
    ]


def fat_deck(size: int = 30) -> list[CardRef]:
    """Oversized deck — draws are diluted."""
    base = silent_starter()
    while len(base) < size:
        base.append(CardRef("STRIKE_SILENT"))
    return base[:size]


# ---------------------------------------------------------------------------
# Expanded scenario set (~60 scenarios across 10 categories)
# ---------------------------------------------------------------------------

def _build_expanded_scenarios() -> list[DeckScenario]:
    out: list[DeckScenario] = []

    # ===================================================================
    # starter_pick — early-game Silent starter decisions (+7)
    # ===================================================================

    out.append(DeckScenario(
        name="starter_take_footwork",
        category="starter_pick",
        description="Silent starter early — Footwork's permanent Dex is a top-tier add",
        state=DeckBuildingState(
            deck=silent_starter(), player=PlayerStats(hp=70, max_hp=70, gold=0),
            act=1, floor=2,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("FOOTWORK")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="starter_take_prepared_v2",
        category="starter_pick",
        description="Silent starter early — Prepared is 0-cost cycle",
        state=DeckBuildingState(
            deck=silent_starter(), player=PlayerStats(hp=65, max_hp=70, gold=0),
            act=1, floor=3,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("PREPARED")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="starter_take_sucker_punch",
        category="starter_pick",
        description="Starter — Sucker Punch is a strong early attack (damage + Weak)",
        state=DeckBuildingState(
            deck=silent_starter(), player=PlayerStats(hp=60, max_hp=70, gold=20),
            act=1, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("SUCKER_PUNCH")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="starter_take_dagger_throw",
        category="starter_pick",
        description="Starter — Dagger Throw is attack + draw, always a solid early pick",
        state=DeckBuildingState(
            deck=silent_starter(), player=PlayerStats(hp=65, max_hp=70, gold=25),
            act=1, floor=2,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DAGGER_THROW")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="starter_take_escape_plan",
        category="starter_pick",
        description="Starter — Escape Plan (0-cost, block + cycle) is excellent",
        state=DeckBuildingState(
            deck=silent_starter(), player=PlayerStats(hp=70, max_hp=70, gold=0),
            act=1, floor=3,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("ESCAPE_PLAN")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="starter_take_acrobatics",
        category="starter_pick",
        description="Starter — Acrobatics cycles 3 and lets you ditch a Defend",
        state=DeckBuildingState(
            deck=silent_starter(), player=PlayerStats(hp=65, max_hp=70, gold=10),
            act=1, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("ACROBATICS")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="starter_take_dagger_spray",
        category="starter_pick",
        description="Starter, multi-enemy floor — Dagger Spray hits all",
        state=DeckBuildingState(
            deck=silent_starter(), player=PlayerStats(hp=55, max_hp=70, gold=30),
            act=1, floor=5,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DAGGER_SPRAY")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    # ===================================================================
    # bloat_skip — skip curses, statuses, redundant basics (+8)
    # ===================================================================

    out.append(DeckScenario(
        name="skip_regret_curse",
        category="bloat_skip",
        description="Offered Regret (curse) — never take",
        state=DeckBuildingState(
            deck=silent_starter(), player=PlayerStats(hp=60, max_hp=70, gold=20),
            act=1, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("REGRET")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1, bad_idx=[0],
    ))

    out.append(DeckScenario(
        name="skip_injury_curse",
        category="bloat_skip",
        description="Offered Injury (curse) — never take",
        state=DeckBuildingState(
            deck=silent_starter(), player=PlayerStats(hp=65, max_hp=70, gold=0),
            act=1, floor=3,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("INJURY")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1, bad_idx=[0],
    ))

    out.append(DeckScenario(
        name="skip_normality_curse",
        category="bloat_skip",
        description="Offered Normality (limits card plays) — always skip",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("FOOTWORK")],
            player=PlayerStats(hp=55, max_hp=70, gold=0),
            act=2, floor=3,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("NORMALITY")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1, bad_idx=[0],
    ))

    out.append(DeckScenario(
        name="skip_dazed_status",
        category="bloat_skip",
        description="Offered Dazed (status, unplayable) — skip",
        state=DeckBuildingState(
            deck=silent_starter(), player=PlayerStats(hp=70, max_hp=70, gold=5),
            act=1, floor=2,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DAZED")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1, bad_idx=[0],
    ))

    out.append(DeckScenario(
        name="heavy_strike_skip_another",
        category="bloat_skip",
        description="Already 8 Strikes — another is pure dilution",
        state=DeckBuildingState(
            deck=heavy_strike_deck(18), player=PlayerStats(hp=50, max_hp=70, gold=60),
            act=2, floor=5,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1, bad_idx=[0],
    ))

    out.append(DeckScenario(
        name="heavy_defend_skip_another",
        category="bloat_skip",
        description="Already 8 Defends — another adds block no better than what you have",
        state=DeckBuildingState(
            deck=heavy_defend_deck(18), player=PlayerStats(hp=55, max_hp=70, gold=50),
            act=2, floor=6,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1, bad_idx=[0],
    ))

    out.append(DeckScenario(
        name="fat_deck_skip_slimed",
        category="bloat_skip",
        description="Fat deck + Slimed — compounding bloat",
        state=DeckBuildingState(
            deck=fat_deck(30), player=PlayerStats(hp=45, max_hp=70, gold=100),
            act=2, floor=10,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("SLIMED")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1, bad_idx=[0],
    ))

    out.append(DeckScenario(
        name="mid_game_skip_strike",
        category="bloat_skip",
        description="Mid-game 22-card deck — an extra Strike dilutes better draws",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("BLADE_DANCE"), CardRef("FOOTWORK"),
                                      CardRef("DEADLY_POISON"), CardRef("NEUTRALIZE"),
                                      CardRef("PREPARED"), CardRef("ACROBATICS"),
                                      CardRef("SUCKER_PUNCH"), CardRef("ESCAPE_PLAN"),
                                      CardRef("DAGGER_THROW"), CardRef("BURST")],
            player=PlayerStats(hp=60, max_hp=70, gold=80),
            act=2, floor=5,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.ADD, card=CardRef("BLADE_DANCE")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=2, bad_idx=[0],
    ))

    # ===================================================================
    # archetype_poison — poison-scaling awareness (+6)
    # ===================================================================

    out.append(DeckScenario(
        name="poison_take_noxious_fumes",
        category="archetype_poison",
        description="Poison deck — Noxious Fumes stacks poison on all enemies each turn",
        state=DeckBuildingState(
            deck=poison_deck(17), player=PlayerStats(hp=60, max_hp=70, gold=60),
            act=1, floor=10,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("NOXIOUS_FUMES")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="poison_take_bouncing_flask",
        category="archetype_poison",
        description="Poison deck — Bouncing Flask spreads poison across enemies",
        state=DeckBuildingState(
            deck=poison_deck(18), player=PlayerStats(hp=55, max_hp=70, gold=80),
            act=2, floor=3,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("BOUNCING_FLASK")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="poison_take_envenom",
        category="archetype_poison",
        description="Poison deck — Envenom adds poison on every attack",
        state=DeckBuildingState(
            deck=poison_deck(18) + [CardRef("DAGGER_THROW")],
            player=PlayerStats(hp=50, max_hp=70, gold=120),
            act=2, floor=8,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("ENVENOM")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="poison_take_nightmare",
        category="archetype_poison",
        description="Poison deck — Nightmare copies Deadly Poison into hand for boss burst",
        state=DeckBuildingState(
            deck=poison_deck(19), player=PlayerStats(hp=50, max_hp=70, gold=150),
            act=2, floor=13,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("NIGHTMARE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="poison_skip_wraith_form",
        category="archetype_poison",
        description="Lean poison deck killing fast — Wraith Form's slow block is off-archetype",
        state=DeckBuildingState(
            deck=poison_deck(18), player=PlayerStats(hp=60, max_hp=70, gold=100),
            act=2, floor=5,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("WRAITH_FORM")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEADLY_POISON")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1, bad_idx=[0],
    ))

    out.append(DeckScenario(
        name="poison_skip_shiv_generator",
        category="archetype_poison",
        description="Poison deck — Infinite Blades's shivs don't stack with poison scaling",
        state=DeckBuildingState(
            deck=poison_deck(19), player=PlayerStats(hp=55, max_hp=70, gold=90),
            act=2, floor=7,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("INFINITE_BLADES")),
            DeckModification(kind=ModKind.ADD, card=CardRef("POISONED_STAB")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
    ))

    # ===================================================================
    # archetype_shiv — shiv-scaling awareness (+6)
    # ===================================================================

    out.append(DeckScenario(
        name="shiv_take_cloak_and_dagger",
        category="archetype_shiv",
        description="Shiv deck — Cloak and Dagger is block + shiv generator",
        state=DeckBuildingState(
            deck=shiv_deck(17), player=PlayerStats(hp=60, max_hp=70, gold=50),
            act=1, floor=9,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("CLOAK_AND_DAGGER")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="shiv_take_storm_of_steel",
        category="archetype_shiv",
        description="Shiv deck — Storm of Steel dumps hand into shivs",
        state=DeckBuildingState(
            deck=shiv_deck(18), player=PlayerStats(hp=55, max_hp=70, gold=80),
            act=2, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("STORM_OF_STEEL")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="shiv_take_dagger_spray",
        category="archetype_shiv",
        description="Shiv deck — Dagger Spray is multi-enemy + cheap",
        state=DeckBuildingState(
            deck=shiv_deck(18), player=PlayerStats(hp=55, max_hp=70, gold=60),
            act=2, floor=6,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DAGGER_SPRAY")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="shiv_take_finisher",
        category="archetype_shiv",
        description="Shiv deck — Finisher scales with attacks played, perfect for shivs",
        state=DeckBuildingState(
            deck=shiv_deck(19), player=PlayerStats(hp=50, max_hp=70, gold=100),
            act=2, floor=10,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("FINISHER")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="shiv_skip_poison_card",
        category="archetype_shiv",
        description="Shiv deck — Deadly Poison is a different archetype",
        state=DeckBuildingState(
            deck=shiv_deck(18), player=PlayerStats(hp=55, max_hp=70, gold=90),
            act=2, floor=5,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DEADLY_POISON")),
            DeckModification(kind=ModKind.ADD, card=CardRef("BLADE_DANCE")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
    ))

    out.append(DeckScenario(
        name="shiv_take_blade_dance_reinforcement",
        category="archetype_shiv",
        description="Shiv deck — another Blade Dance is archetype reinforcement",
        state=DeckBuildingState(
            deck=shiv_deck(18), player=PlayerStats(hp=60, max_hp=70, gold=40),
            act=1, floor=13,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("BLADE_DANCE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    # ===================================================================
    # archetype_cycle — draw/discard/cycle (+7)
    # ===================================================================

    out.append(DeckScenario(
        name="cycle_take_adrenaline",
        category="archetype_cycle",
        description="Cycle deck — Adrenaline is free energy + draw",
        state=DeckBuildingState(
            deck=cycle_deck(18), player=PlayerStats(hp=60, max_hp=70, gold=70),
            act=2, floor=3,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("ADRENALINE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="cycle_take_backflip",
        category="archetype_cycle",
        description="Cycle deck — Backflip is block + draw 2",
        state=DeckBuildingState(
            deck=cycle_deck(17), player=PlayerStats(hp=65, max_hp=70, gold=40),
            act=1, floor=7,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("BACKFLIP")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="cycle_take_calculated_gamble",
        category="archetype_cycle",
        description="Cycle deck with cycle payoff — Calculated Gamble is pure engine",
        state=DeckBuildingState(
            deck=cycle_deck(18), player=PlayerStats(hp=55, max_hp=70, gold=60),
            act=2, floor=5,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("CALCULATED_GAMBLE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="cycle_take_distraction",
        category="archetype_cycle",
        description="Cycle deck — Distraction generates random 0-cost cards (draw engine)",
        state=DeckBuildingState(
            deck=cycle_deck(18), player=PlayerStats(hp=60, max_hp=70, gold=30),
            act=2, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DISTRACTION")),
            DeckModification(kind=ModKind.ADD, card=CardRef("SLIMED")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0, bad_idx=[1],
    ))

    out.append(DeckScenario(
        name="cycle_take_bullet_time",
        category="archetype_cycle",
        description="Cycle deck late — Bullet Time enables 0-cost hand plays",
        state=DeckBuildingState(
            deck=cycle_deck(19), player=PlayerStats(hp=55, max_hp=70, gold=100),
            act=2, floor=10,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("BULLET_TIME")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="cycle_take_burst",
        category="archetype_cycle",
        description="Cycle deck — Burst doubles a skill play, combos with cycle cards",
        state=DeckBuildingState(
            deck=cycle_deck(18), player=PlayerStats(hp=60, max_hp=70, gold=60),
            act=1, floor=12,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("BURST")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="cycle_skip_heavy_attack",
        category="archetype_cycle",
        description="Thin cycle deck — big slow attack ruins the lean curve",
        state=DeckBuildingState(
            deck=cycle_deck(16), player=PlayerStats(hp=60, max_hp=70, gold=80),
            act=2, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("SKEWER")),
            DeckModification(kind=ModKind.ADD, card=CardRef("PREPARED")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
    ))

    # ===================================================================
    # archetype_block — block/durability archetype (+6)
    # ===================================================================

    out.append(DeckScenario(
        name="block_take_footwork",
        category="archetype_block",
        description="Block-leaning deck — Footwork's permanent Dex scales all block",
        state=DeckBuildingState(
            deck=block_focused_deck(17), player=PlayerStats(hp=55, max_hp=70, gold=50),
            act=2, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("FOOTWORK")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="block_take_blur",
        category="archetype_block",
        description="Block deck — Blur keeps block between turns",
        state=DeckBuildingState(
            deck=block_focused_deck(17), player=PlayerStats(hp=45, max_hp=70, gold=60),
            act=2, floor=7,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("BLUR")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="block_take_wraith_form",
        category="archetype_block",
        description="High-durability deck, act 2 elites — Wraith Form's Intangible is huge",
        state=DeckBuildingState(
            deck=block_focused_deck(19) + [CardRef("FOOTWORK")],
            player=PlayerStats(hp=55, max_hp=70, gold=150),
            act=2, floor=9,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("WRAITH_FORM")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="block_take_reflex",
        category="archetype_block",
        description="Block deck with Cloak/Dagger — Reflex off-discard synergy",
        state=DeckBuildingState(
            deck=block_focused_deck(17) + [CardRef("CLOAK_AND_DAGGER")],
            player=PlayerStats(hp=50, max_hp=70, gold=40),
            act=2, floor=5,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("REFLEX")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="block_take_dodge_and_roll",
        category="archetype_block",
        description="Block deck — Dodge and Roll cycles block between turns",
        state=DeckBuildingState(
            deck=block_focused_deck(17), player=PlayerStats(hp=55, max_hp=70, gold=40),
            act=1, floor=10,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DODGE_AND_ROLL")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="block_skip_poison",
        category="archetype_block",
        description="Block-survival deck — poison is a different archetype",
        state=DeckBuildingState(
            deck=block_focused_deck(18), player=PlayerStats(hp=50, max_hp=70, gold=80),
            act=2, floor=6,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DEADLY_POISON")),
            DeckModification(kind=ModKind.ADD, card=CardRef("BLUR")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
    ))

    # ===================================================================
    # deck_size — thin vs fat tradeoffs (+4)
    # ===================================================================

    out.append(DeckScenario(
        name="thin_deck_take_omnislice",
        category="deck_size",
        description="Thin deck — Omnislice is a big single-card payoff",
        state=DeckBuildingState(
            deck=exhaust_thin_deck(), player=PlayerStats(hp=50, max_hp=70, gold=120),
            act=2, floor=8,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("OMNISLICE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="fat_deck_skip_average_add",
        category="deck_size",
        description="25+-card deck — only premium adds justify further bloat",
        state=DeckBuildingState(
            deck=fat_deck(27) + [CardRef("BLADE_DANCE")],
            player=PlayerStats(hp=45, max_hp=70, gold=100),
            act=2, floor=9,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DAGGER_THROW")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
    ))

    out.append(DeckScenario(
        name="thin_deck_skip_defender",
        category="deck_size",
        description="Thin aggressive deck — extra Defend slows the curve",
        state=DeckBuildingState(
            deck=[CardRef("STRIKE_SILENT")] * 4 + [CardRef("BLADE_DANCE")] * 3
                 + [CardRef("DEADLY_POISON"), CardRef("NEUTRALIZE"),
                    CardRef("SURVIVOR"), CardRef("FOOTWORK"), CardRef("PREPARED")],
            player=PlayerStats(hp=60, max_hp=70, gold=60),
            act=2, floor=5,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.ADD, card=CardRef("BLADE_DANCE")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
    ))

    out.append(DeckScenario(
        name="thin_deck_take_bullet_time_combo",
        category="deck_size",
        description="Thin deck — Bullet Time is a big 0-cost combo turn",
        state=DeckBuildingState(
            deck=exhaust_thin_deck() + [CardRef("BLADE_DANCE"), CardRef("NEUTRALIZE")],
            player=PlayerStats(hp=50, max_hp=70, gold=100),
            act=3, floor=2,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("BULLET_TIME")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    # ===================================================================
    # scaling_power — Power cards for long combats (+6)
    # ===================================================================

    out.append(DeckScenario(
        name="early_act_take_well_laid_plans",
        category="scaling_power",
        description="Early-mid — Well-Laid Plans retains cards between turns",
        state=DeckBuildingState(
            deck=cycle_deck(17), player=PlayerStats(hp=60, max_hp=70, gold=60),
            act=1, floor=11,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("WELL_LAID_PLANS")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="mid_act_take_accuracy_general",
        category="scaling_power",
        description="Mid-game deck — Accuracy scales shivs + attack count",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("BLADE_DANCE"), CardRef("DAGGER_THROW"),
                                      CardRef("DAGGER_SPRAY"), CardRef("CLOAK_AND_DAGGER"),
                                      CardRef("SUCKER_PUNCH")],
            player=PlayerStats(hp=55, max_hp=70, gold=80),
            act=2, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("ACCURACY")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="mid_act_take_hidden_daggers",
        category="scaling_power",
        description="Mid-game deck — Hidden Daggers generates shivs over time",
        state=DeckBuildingState(
            deck=shiv_deck(17), player=PlayerStats(hp=55, max_hp=70, gold=70),
            act=2, floor=6,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("HIDDEN_DAGGERS")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="late_act3_skip_slow_power",
        category="scaling_power",
        description="Act 3 floor 15, boss next — slow Power has no time to pay off",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("FOOTWORK"), CardRef("BLADE_DANCE")] * 2
                 + [CardRef("DEADLY_POISON"), CardRef("ACCURACY")],
            player=PlayerStats(hp=40, max_hp=70, gold=80),
            act=3, floor=15, boss_id="CORRUPT_HEART",
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("ENVENOM")),
            DeckModification(kind=ModKind.ADD, card=CardRef("FINISHER")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=2,
    ))

    out.append(DeckScenario(
        name="mid_act_take_infinite_blades_any",
        category="scaling_power",
        description="Mid-game — Infinite Blades creates shiv every turn, scales well",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("BLADE_DANCE")] * 2 + [CardRef("ACCURACY")],
            player=PlayerStats(hp=60, max_hp=70, gold=90),
            act=2, floor=5,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("INFINITE_BLADES")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="heavy_powers_skip_more",
        category="scaling_power",
        description="5 Powers already — another is redundant and slow",
        state=DeckBuildingState(
            deck=heavy_powers_deck(20), player=PlayerStats(hp=50, max_hp=70, gold=120),
            act=2, floor=7,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("WELL_LAID_PLANS")),
            DeckModification(kind=ModKind.ADD, card=CardRef("BLADE_DANCE")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
    ))

    # ===================================================================
    # late_game_hp — low HP affects picks (+5)
    # ===================================================================

    out.append(DeckScenario(
        name="low_hp_take_blur",
        category="late_game_hp",
        description="Low HP act 2 — Blur gives immediate block + keeps between turns",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("FOOTWORK"), CardRef("BLADE_DANCE")],
            player=PlayerStats(hp=15, max_hp=70, gold=80),
            act=2, floor=8,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("BLUR")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="low_hp_take_survivor",
        category="late_game_hp",
        description="Low HP — Survivor is block + discard utility",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("DEADLY_POISON")],
            player=PlayerStats(hp=20, max_hp=70, gold=40),
            act=2, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("SURVIVOR")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="low_hp_skip_slow_power",
        category="late_game_hp",
        description="Low HP with boss imminent — no time for slow setup",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("BLADE_DANCE")],
            player=PlayerStats(hp=15, max_hp=70, gold=120),
            act=2, floor=15, boss_id="AUTOMATON",
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("ENVENOM")),
            DeckModification(kind=ModKind.ADD, card=CardRef("BLUR")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1,
    ))

    out.append(DeckScenario(
        name="low_hp_skip_slimed_always",
        category="late_game_hp",
        description="Low HP, 30 HP in act 3 — absolutely never take curses",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("FOOTWORK"), CardRef("PREPARED"),
                                     CardRef("DEADLY_POISON")],
            player=PlayerStats(hp=25, max_hp=70, gold=80),
            act=3, floor=3,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("SLIMED")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=1, bad_idx=[0],
    ))

    out.append(DeckScenario(
        name="low_hp_take_footwork_starter",
        category="late_game_hp",
        description="Low HP early starter — Footwork scales Defend block",
        state=DeckBuildingState(
            deck=silent_starter(),
            player=PlayerStats(hp=25, max_hp=70, gold=40),
            act=1, floor=7,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("FOOTWORK")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    # ===================================================================
    # utility — general-purpose strong picks (+4)
    # ===================================================================

    out.append(DeckScenario(
        name="offered_neutralize_weak",
        category="utility",
        description="Any Silent deck — Neutralize (+Weak) is always a solid add",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("BLADE_DANCE"), CardRef("FOOTWORK")],
            player=PlayerStats(hp=60, max_hp=70, gold=40),
            act=2, floor=4,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("NEUTRALIZE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="offered_malaise_elite",
        category="utility",
        description="Pre-elite — Malaise is strong elite debuff",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("BLADE_DANCE"), CardRef("DEADLY_POISON")],
            player=PlayerStats(hp=55, max_hp=70, gold=80),
            act=1, floor=8,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("MALAISE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("STRIKE_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="offered_expose_general",
        category="utility",
        description="Silent deck — Expose is Vulnerable + exhaust on low cost",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("FOOTWORK")],
            player=PlayerStats(hp=60, max_hp=70, gold=30),
            act=1, floor=9,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("EXPOSE")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    out.append(DeckScenario(
        name="offered_piercing_wail_multi",
        category="utility",
        description="Multi-enemy floors — Piercing Wail AoE weakens",
        state=DeckBuildingState(
            deck=silent_starter() + [CardRef("BLADE_DANCE")],
            player=PlayerStats(hp=50, max_hp=70, gold=40),
            act=2, floor=3,
        ),
        candidates=[
            DeckModification(kind=ModKind.ADD, card=CardRef("PIERCING_WAIL")),
            DeckModification(kind=ModKind.ADD, card=CardRef("DEFEND_SILENT")),
            DeckModification(kind=ModKind.IDENTITY),
        ],
        best_idx=0,
    ))

    return out


# ---------------------------------------------------------------------------
# Eval harness
# ---------------------------------------------------------------------------

def run_eval(net: DeckNet, card_vocab: dict[str, int], verbose: bool = False) -> dict:
    """Run all scenarios, return pass/fail summary.

    Returns dict with:
      passed: int
      total: int
      score: float  (passed/total)
      results: list of {name, passed, chosen, best, probs} dicts
    """
    scenarios = build_scenarios()
    net_eval = net.eval()

    by_category_idx: dict[str, list[bool]] = {}
    results: list[dict] = []

    with torch.no_grad():
        for sc in scenarios:
            # Build candidate post-mod states
            candidate_states = []
            for mod in sc.candidates:
                try:
                    candidate_states.append(apply_mod(sc.state, mod))
                except Exception:
                    candidate_states.append(sc.state)  # safe fallback

            # Batch-evaluate V on all candidates
            batch = encode_batch(candidate_states, card_vocab)
            values = net_eval(
                batch["card_ids"], batch["card_stats"],
                batch["deck_mask"], batch["global_state"],
            )
            best = int(torch.argmax(values).item())
            passed = (best == sc.best_idx) and (best not in sc.bad_idx)

            results.append({
                "name": sc.name,
                "description": sc.description,
                "passed": passed,
                "chosen": best,
                "best": sc.best_idx,
                "bad": sc.bad_idx,
                "values": [round(v, 4) for v in values.tolist()],
            })

            if verbose:
                status = "ok" if passed else "MISS"
                if best in sc.bad_idx:
                    status = "BAD"
                print(f"  {status:>4}  {sc.name}")
                if not passed:
                    print(f"         chose idx={best} (V={values[best].item():+.3f}), "
                          f"best was idx={sc.best_idx} (V={values[sc.best_idx].item():+.3f})")
                    print(f"         values: {[f'{v:+.3f}' for v in values.tolist()]}")

    passed_count = sum(1 for r in results if r["passed"])
    return {
        "passed": passed_count,
        "total": len(results),
        "score": round(passed_count / max(len(results), 1), 4),
        "results": results,
    }
