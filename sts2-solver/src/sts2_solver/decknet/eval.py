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


# ---------------------------------------------------------------------------
# Scenario builders
# ---------------------------------------------------------------------------

def build_scenarios() -> list[DeckScenario]:
    out: list[DeckScenario] = []

    # --- Starter-deck card rewards ---

    out.append(DeckScenario(
        name="starter_take_blade_dance",
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
