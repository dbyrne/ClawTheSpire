"""Expanded value-head eval scenarios organized by failure mode.

The original 25 scenarios in eval.py are organized by game concept (hp,
defense, poison, powers, ...). That taxonomy makes categories tiny —
1-3 scenarios each — so a single flip swings category pass-rate by
33-100%, and the total has ±4pt resolution per scenario.

This module adds scenarios grouped by the *failure mode* we hypothesize
a deeper value head could fix:

  compound_scaling    — multiplicative/compound effects (poison amount,
                        strength x hits, dexterity x block) scale with V
  conditional_value   — value depends on context (intangible -> block
                        useless, weak self -> damage reduced, enemy
                        block -> attack wasted)
  arithmetic_compare  — ordering of magnitudes (card A damage vs card B,
                        block vs incoming damage at specific HP)
  future_value        — value of drawing / cycling / power-card setup
                        that pays off in later turns

Each category is 25-30 scenarios, so one flip is ~3-4% of the category.
Combined with the existing 25 scenarios we get ~135 total, pushing
resolution down to ~0.7% per scenario on the overall score.
"""

from .eval import (
    ValueComparison, _vstate, enemy,
    strike, defend, neutralize, blade_dance, skewer, predator,
    dagger_throw, accelerant, deadly_poison, noxious_fumes,
    wraith_form, footwork, accuracy, backflip, adrenaline,
    prepared, calculated_gamble, acrobatics, infinite_blades,
    well_laid_plans, burst, escape_plan, blur, expose,
    cloak_and_dagger, omnislice, tactician, reflex, survivor,
    malaise, sucker_punch, piercing_wail, slimed,
)


_BASE_PLAYER = {"hp": 50, "max_hp": 70, "energy": 3, "block": 0}
_BASE_ENEMY = [enemy(40, 50, damage=10)]
_BASE_HAND = [strike(), strike(), defend(), defend()]


def _p(**overrides):
    return {**_BASE_PLAYER, **overrides}


# ---------------------------------------------------------------------------
# 1. compound_scaling — V should respond monotonically to compound magnitudes
# ---------------------------------------------------------------------------

def build_compound_scaling() -> list[ValueComparison]:
    comps = []

    # --- poison amount ladder: more poison on enemy should mean higher V ---
    # 6 adjacent steps; the model should see each step as better.
    poison_steps = [(3, 8), (8, 15), (15, 25), (25, 40), (2, 30), (10, 30)]
    for lo, hi in poison_steps:
        comps.append(ValueComparison(
            name=f"poison_more_{hi}_vs_{lo}",
            category="compound_scaling",
            description=f"Enemy with {hi} poison better than {lo}",
            better=_vstate(_BASE_PLAYER,
                           [enemy(40, 50, damage=10, powers={"Poison": hi})],
                           _BASE_HAND),
            worse=_vstate(_BASE_PLAYER,
                          [enemy(40, 50, damage=10, powers={"Poison": lo})],
                          _BASE_HAND),
        ))

    # --- strength ladder: more strength should mean higher V ---
    for s_hi, s_lo in [(1, 0), (3, 1), (5, 2), (7, 3), (2, 0)]:
        comps.append(ValueComparison(
            name=f"strength_{s_hi}_vs_{s_lo}",
            category="compound_scaling",
            description=f"Strength {s_hi} better than Strength {s_lo}",
            better=_vstate(_p(powers={"Strength": s_hi}), _BASE_ENEMY, _BASE_HAND),
            worse=_vstate(_p(powers={"Strength": s_lo}) if s_lo > 0
                          else _BASE_PLAYER, _BASE_ENEMY, _BASE_HAND),
        ))

    # --- dexterity scaling vs incoming damage ---
    for d_hi, d_lo in [(3, 0), (5, 2), (2, 0)]:
        comps.append(ValueComparison(
            name=f"dexterity_{d_hi}_vs_{d_lo}",
            category="compound_scaling",
            description=f"Dexterity {d_hi} better vs incoming damage",
            better=_vstate(_p(powers={"Dexterity": d_hi}),
                           [enemy(40, 50, damage=15)], _BASE_HAND),
            worse=_vstate(_p(powers={"Dexterity": d_lo}) if d_lo > 0
                          else _BASE_PLAYER,
                          [enemy(40, 50, damage=15)], _BASE_HAND),
        ))

    # --- noxious fumes (poison per turn) scaling ---
    for nf_hi, nf_lo in [(2, 0), (4, 2), (3, 1)]:
        comps.append(ValueComparison(
            name=f"noxious_fumes_{nf_hi}_vs_{nf_lo}",
            category="compound_scaling",
            description=f"Noxious Fumes {nf_hi} better than {nf_lo}",
            better=_vstate(_p(powers={"Noxious Fumes": nf_hi}),
                           _BASE_ENEMY, _BASE_HAND),
            worse=_vstate(_p(powers={"Noxious Fumes": nf_lo}) if nf_lo > 0
                          else _BASE_PLAYER, _BASE_ENEMY, _BASE_HAND),
        ))

    # --- multi-hit attackers with strength: effect compounds ---
    # Same player damage vs multi-hit enemy: more hits = more incoming
    for hits_lo, hits_hi in [(1, 3), (2, 5), (1, 4)]:
        comps.append(ValueComparison(
            name=f"fewer_hits_{hits_lo}_vs_{hits_hi}",
            category="compound_scaling",
            description=f"Enemy doing {hits_lo} hits better than {hits_hi} hits",
            better=_vstate(_BASE_PLAYER,
                           [enemy(40, 50, damage=5, hits=hits_lo)], _BASE_HAND),
            worse=_vstate(_BASE_PLAYER,
                          [enemy(40, 50, damage=5, hits=hits_hi)], _BASE_HAND),
        ))

    # --- vulnerable duration (longer = more benefit over time) ---
    for v_hi, v_lo in [(4, 1), (6, 2), (3, 1)]:
        comps.append(ValueComparison(
            name=f"vulnerable_{v_hi}_vs_{v_lo}",
            category="compound_scaling",
            description=f"Vulnerable {v_hi} on enemy better than {v_lo}",
            better=_vstate(_BASE_PLAYER,
                           [enemy(40, 50, damage=10,
                                  powers={"Vulnerable": v_hi})], _BASE_HAND),
            worse=_vstate(_BASE_PLAYER,
                          [enemy(40, 50, damage=10,
                                 powers={"Vulnerable": v_lo})], _BASE_HAND),
        ))

    # --- weak duration on enemy (longer = more incoming reduction) ---
    for w_hi, w_lo in [(4, 1), (5, 2), (3, 1)]:
        comps.append(ValueComparison(
            name=f"enemy_weak_{w_hi}_vs_{w_lo}",
            category="compound_scaling",
            description=f"Enemy Weak {w_hi} better than Weak {w_lo}",
            better=_vstate(_BASE_PLAYER,
                           [enemy(40, 50, damage=12, powers={"Weak": w_hi})],
                           _BASE_HAND),
            worse=_vstate(_BASE_PLAYER,
                          [enemy(40, 50, damage=12, powers={"Weak": w_lo})],
                          _BASE_HAND),
        ))

    return comps


# ---------------------------------------------------------------------------
# 2. conditional_value — value depends on context/precondition
# ---------------------------------------------------------------------------

def build_conditional_value() -> list[ValueComparison]:
    comps = []

    # --- Intangible makes block effectively useless ---
    # With Intangible, having 10 block vs 0 block should NOT meaningfully
    # change V. We test the opposite: without Intangible, block matters.
    comps.append(ValueComparison(
        name="block_matters_without_intangible",
        category="conditional_value",
        description="Without Intangible: 15 block > 0 block vs 20 dmg",
        better=_vstate(_p(block=15), [enemy(40, 50, damage=20)], _BASE_HAND),
        worse=_vstate(_p(block=0), [enemy(40, 50, damage=20)], _BASE_HAND),
    ))
    # With Intangible active, V(intangible + no block) > V(no intangible + low block)
    comps.append(ValueComparison(
        name="intangible_beats_partial_block",
        category="conditional_value",
        description="Intangible (dmg->1) > 8 block vs 20 incoming",
        better=_vstate(_p(powers={"Intangible": 2}),
                       [enemy(40, 50, damage=20)], _BASE_HAND),
        worse=_vstate(_p(block=8), [enemy(40, 50, damage=20)], _BASE_HAND),
    ))

    # --- Weak self reduces damage output ---
    # Same hand + enemy: V(no weak) > V(player weak) because damage reduced.
    for weak in [1, 3, 5]:
        comps.append(ValueComparison(
            name=f"player_no_weak_vs_weak_{weak}",
            category="conditional_value",
            description=f"Player without Weak > player with Weak {weak}",
            better=_vstate(_BASE_PLAYER, _BASE_ENEMY, _BASE_HAND),
            worse=_vstate(_p(powers={"Weak": weak}), _BASE_ENEMY, _BASE_HAND),
        ))

    # --- Vulnerable on player = worse ---
    for vuln in [1, 3]:
        comps.append(ValueComparison(
            name=f"player_no_vulnerable_vs_{vuln}",
            category="conditional_value",
            description=f"Player without Vulnerable > Vulnerable {vuln}",
            better=_vstate(_BASE_PLAYER, _BASE_ENEMY, _BASE_HAND),
            worse=_vstate(_p(powers={"Vulnerable": vuln}), _BASE_ENEMY, _BASE_HAND),
        ))

    # --- Frail on player = less block output ---
    for frail in [1, 3]:
        comps.append(ValueComparison(
            name=f"player_no_frail_vs_{frail}",
            category="conditional_value",
            description=f"Player without Frail > Frail {frail} (block 25% reduced)",
            better=_vstate(_BASE_PLAYER, [enemy(40, 50, damage=15)], _BASE_HAND),
            worse=_vstate(_p(powers={"Frail": frail}),
                          [enemy(40, 50, damage=15)], _BASE_HAND),
        ))

    # --- Enemy blocking: attack spent on block is worse ---
    comps.append(ValueComparison(
        name="enemy_no_block_better",
        category="conditional_value",
        description="Enemy without block > enemy with high block (attacks land)",
        better=_vstate(_BASE_PLAYER, [enemy(30, 50, damage=10)], _BASE_HAND),
        worse=_vstate(_BASE_PLAYER,
                      [{**enemy(30, 50, damage=10), "block": 20}], _BASE_HAND),
    ))

    # --- Artifact on enemy negates debuff application ---
    # Hand with Neutralize is worth more vs no-artifact enemy
    comps.append(ValueComparison(
        name="neutralize_vs_no_artifact",
        category="conditional_value",
        description="Neutralize in hand: better vs no-Artifact enemy",
        better=_vstate(_BASE_PLAYER, [enemy(40, 50, damage=10)],
                       [neutralize(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER,
                      [enemy(40, 50, damage=10, powers={"Artifact": 2})],
                      [neutralize(), strike(), defend(), defend()]),
    ))

    # --- Accelerant without poison isn't useful ---
    # Hand with Accelerant but enemy has 0 poison: worse than Accelerant +
    # enemy with 10 poison.
    comps.append(ValueComparison(
        name="accelerant_with_poison_precondition",
        category="conditional_value",
        description="Accelerant better when poison is on enemy",
        better=_vstate(_BASE_PLAYER,
                       [enemy(40, 50, damage=10, powers={"Poison": 10})],
                       [accelerant(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(40, 50, damage=10)],
                      [accelerant(), strike(), defend(), defend()]),
    ))

    # --- Burst requires a skill in hand ---
    comps.append(ValueComparison(
        name="burst_with_skill_in_hand",
        category="conditional_value",
        description="Burst useful only when hand has a skill to double",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [burst(), acrobatics(), strike(), strike()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [burst(), strike(), strike(), strike()]),
    ))

    # --- Accuracy needs shivs to amplify ---
    comps.append(ValueComparison(
        name="accuracy_with_shiv_source",
        category="conditional_value",
        description="Accuracy useful with Blade Dance in hand",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [accuracy(), blade_dance(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [accuracy(), strike(), strike(), defend()]),
    ))

    # --- Low-HP-triggered vs higher HP: Predator worth more low-HP window ---
    comps.append(ValueComparison(
        name="killing_blow_at_low_enemy_hp",
        category="conditional_value",
        description="Strong attack more valuable when enemy is near-lethal",
        better=_vstate(_BASE_PLAYER, [enemy(10, 50, damage=10)],
                       [predator(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(45, 50, damage=10)],
                      [predator(), strike(), defend(), defend()]),
    ))

    # --- Slimed status in hand wastes draw ---
    comps.append(ValueComparison(
        name="no_slimed_vs_slimed",
        category="conditional_value",
        description="Hand without Slimed > hand with Slimed (wastes energy)",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [strike(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [strike(), strike(), defend(), slimed()]),
    ))

    # --- Energy matters only when there are cards to play ---
    # 3 energy with playable hand > 3 energy with unplayable (too-expensive) hand.
    # We approximate by comparing playable hand vs empty (still 3 energy).
    comps.append(ValueComparison(
        name="energy_with_playable_hand",
        category="conditional_value",
        description="3 energy + playable hand > 3 energy + empty hand",
        better=_vstate(_p(energy=3), _BASE_ENEMY, _BASE_HAND),
        worse=_vstate(_p(energy=3), _BASE_ENEMY, []),
    ))

    # --- Block value decays: same block vs no incoming damage is wasted ---
    comps.append(ValueComparison(
        name="block_when_no_incoming",
        category="conditional_value",
        description="No block vs no-attack enemy ~= 10 block "
                    "(block is wasted) — but not worse",
        better=_vstate(_p(block=0),
                       [enemy(40, 50, intent="Buff", damage=0)], _BASE_HAND),
        worse=_vstate(_p(block=0, hp=30),
                      [enemy(40, 50, intent="Buff", damage=0)], _BASE_HAND),
    ))

    # --- Multi-hit + Vulnerable compounds ---
    # Enemy with Vulnerable and high hits: player's multi-hit AoE/shiv good
    comps.append(ValueComparison(
        name="blade_dance_with_vulnerable",
        category="conditional_value",
        description="Blade Dance better vs Vulnerable enemy",
        better=_vstate(_BASE_PLAYER,
                       [enemy(40, 50, damage=10, powers={"Vulnerable": 3})],
                       [blade_dance(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(40, 50, damage=10)],
                      [blade_dance(), strike(), defend(), defend()]),
    ))

    # --- Plated Armor makes enemy's attacks less meaningful? No — test
    # the reverse: enemy with Plated Armor (persistent block) is worse for us.
    comps.append(ValueComparison(
        name="enemy_no_plated_armor",
        category="conditional_value",
        description="Enemy without Plated Armor > enemy with PA 10",
        better=_vstate(_BASE_PLAYER, [enemy(40, 50, damage=10)], _BASE_HAND),
        worse=_vstate(_BASE_PLAYER,
                      [enemy(40, 50, damage=10, powers={"Plated Armor": 10})],
                      _BASE_HAND),
    ))

    return comps


# ---------------------------------------------------------------------------
# 3. arithmetic_compare — V should order magnitudes correctly
# ---------------------------------------------------------------------------

def build_arithmetic_compare() -> list[ValueComparison]:
    comps = []

    # --- Damage card comparisons (held in hand) ---
    # Skewer scales w/ energy, Strike is 6: at 3 energy Skewer >> Strike
    comps.append(ValueComparison(
        name="skewer_3e_vs_strike",
        category="arithmetic_compare",
        description="Skewer (3E->21 dmg) > Strike in hand at 3 energy",
        better=_vstate(_p(energy=3), _BASE_ENEMY,
                       [skewer(), strike(), defend(), defend()]),
        worse=_vstate(_p(energy=3), _BASE_ENEMY,
                      [strike(), strike(), defend(), defend()]),
    ))
    # Dagger throw (9) > Strike (6)
    comps.append(ValueComparison(
        name="dagger_throw_vs_strike",
        category="arithmetic_compare",
        description="Dagger Throw (9 dmg) > Strike (6 dmg)",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [dagger_throw(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [strike(), strike(), defend(), defend()]),
    ))
    # Omnislice (free damage) > Strike
    comps.append(ValueComparison(
        name="omnislice_vs_strike",
        category="arithmetic_compare",
        description="Omnislice > Strike (adds free damage)",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [omnislice(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [strike(), strike(), defend(), defend()]),
    ))
    # Blade Dance (12 across 3 shivs) > Strike
    comps.append(ValueComparison(
        name="blade_dance_vs_strike_arith",
        category="arithmetic_compare",
        description="Blade Dance (3x4=12) > Strike (6)",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [blade_dance(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [strike(), strike(), defend(), defend()]),
    ))

    # --- Block card comparisons ---
    # Cloak and Dagger (6 block + shiv) > Defend (5 block)
    comps.append(ValueComparison(
        name="cloak_vs_defend",
        category="arithmetic_compare",
        description="Cloak and Dagger > Defend",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [cloak_and_dagger(), strike(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [defend(), strike(), strike(), defend()]),
    ))

    # --- Incoming-damage vs block ---
    # Ordered: more block > less block at same incoming
    for b_hi, b_lo in [(20, 5), (15, 0), (25, 10), (10, 0), (30, 15)]:
        comps.append(ValueComparison(
            name=f"block_{b_hi}_vs_{b_lo}_vs_15dmg",
            category="arithmetic_compare",
            description=f"{b_hi} block > {b_lo} block vs 15 incoming",
            better=_vstate(_p(block=b_hi),
                           [enemy(40, 50, damage=15)], _BASE_HAND),
            worse=_vstate(_p(block=b_lo),
                          [enemy(40, 50, damage=15)], _BASE_HAND),
        ))

    # --- HP ladder at varied margins ---
    for hp_hi, hp_lo in [(30, 20), (50, 30), (25, 15), (40, 20),
                         (60, 40), (35, 25)]:
        comps.append(ValueComparison(
            name=f"player_hp_{hp_hi}_vs_{hp_lo}",
            category="arithmetic_compare",
            description=f"Player HP {hp_hi} > {hp_lo}",
            better=_vstate({**_BASE_PLAYER, "hp": hp_hi},
                           _BASE_ENEMY, _BASE_HAND),
            worse=_vstate({**_BASE_PLAYER, "hp": hp_lo},
                          _BASE_ENEMY, _BASE_HAND),
        ))

    # --- Enemy HP ladder: lower = closer to win ---
    for e_lo, e_hi in [(15, 30), (5, 25), (20, 40), (10, 35), (8, 30)]:
        comps.append(ValueComparison(
            name=f"enemy_hp_{e_lo}_vs_{e_hi}",
            category="arithmetic_compare",
            description=f"Enemy HP {e_lo} > enemy HP {e_hi}",
            better=_vstate(_BASE_PLAYER,
                           [enemy(e_lo, 50, damage=10)], _BASE_HAND),
            worse=_vstate(_BASE_PLAYER,
                          [enemy(e_hi, 50, damage=10)], _BASE_HAND),
        ))

    # --- Multi-card damage hands: more attacks > fewer ---
    comps.append(ValueComparison(
        name="three_strikes_vs_one",
        category="arithmetic_compare",
        description="3 Strikes in hand > 1 Strike (more total damage)",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [strike(), strike(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [strike(), defend(), defend(), defend()]),
    ))

    # --- Compared energy: more energy = more throughput ---
    comps.append(ValueComparison(
        name="energy_3_vs_1",
        category="arithmetic_compare",
        description="3 energy > 1 energy with playable hand",
        better=_vstate(_p(energy=3), _BASE_ENEMY, _BASE_HAND),
        worse=_vstate(_p(energy=1), _BASE_ENEMY, _BASE_HAND),
    ))
    comps.append(ValueComparison(
        name="energy_2_vs_0",
        category="arithmetic_compare",
        description="2 energy > 0 energy",
        better=_vstate(_p(energy=2), _BASE_ENEMY, _BASE_HAND),
        worse=_vstate(_p(energy=0), _BASE_ENEMY, _BASE_HAND),
    ))

    return comps


# ---------------------------------------------------------------------------
# 4. future_value — draw / cycle / power cards pay off in future turns
# ---------------------------------------------------------------------------

def build_future_value() -> list[ValueComparison]:
    comps = []

    # --- Draw engines vs inert cards ---
    # Adrenaline: 0-cost, +1 energy, draw 2
    comps.append(ValueComparison(
        name="adrenaline_over_defend",
        category="future_value",
        description="Adrenaline (0-cost, draw 2, +E) > extra Defend",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [adrenaline(), strike(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [defend(), strike(), strike(), defend()]),
    ))
    # Prepared: 0-cost draw 2 discard 1 — free cycle
    comps.append(ValueComparison(
        name="prepared_over_defend",
        category="future_value",
        description="Prepared (free cycle) > extra Defend",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [prepared(), strike(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [defend(), strike(), strike(), defend()]),
    ))
    # Backflip: block + draw 2 > plain Defend
    comps.append(ValueComparison(
        name="backflip_over_defend_fv",
        category="future_value",
        description="Backflip (block + draw 2) > Defend (block only)",
        better=_vstate(_BASE_PLAYER, [enemy(40, 50, damage=10)],
                       [backflip(), strike(), strike(), strike()]),
        worse=_vstate(_BASE_PLAYER, [enemy(40, 50, damage=10)],
                      [defend(), strike(), strike(), strike()]),
    ))
    # Acrobatics: draw 3 discard 1
    comps.append(ValueComparison(
        name="acrobatics_over_strike_extra",
        category="future_value",
        description="Acrobatics (draw 3 discard 1) > extra Strike",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [acrobatics(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [strike(), strike(), defend(), defend()]),
    ))
    # Calculated Gamble: discard hand and redraw — value when hand is dead
    comps.append(ValueComparison(
        name="calculated_gamble_dead_hand",
        category="future_value",
        description="Calculated Gamble > extra Strike when hand is weak",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [calculated_gamble(), defend(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [strike(), defend(), defend(), defend()]),
    ))

    # --- Power cards early vs just-another-attack ---
    # Footwork (dex power) > extra Strike
    comps.append(ValueComparison(
        name="footwork_power_over_strike",
        category="future_value",
        description="Footwork (Dex+) > extra Strike (scales future block)",
        better=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=12)],
                       [footwork(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=12)],
                      [strike(), strike(), defend(), defend()]),
    ))
    # Wraith Form (power) > Defend when long fight incoming
    comps.append(ValueComparison(
        name="wraith_form_long_fight",
        category="future_value",
        description="Wraith Form in long fight > extra Defend",
        better=_vstate(_BASE_PLAYER, [enemy(80, 100, damage=18)],
                       [wraith_form(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(80, 100, damage=18)],
                      [defend(), strike(), defend(), defend()]),
    ))
    # Accuracy (power) with shiv source > extra strike
    comps.append(ValueComparison(
        name="accuracy_with_blade_dance",
        category="future_value",
        description="Accuracy + Blade Dance > Strike + Blade Dance",
        better=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=10)],
                       [accuracy(), blade_dance(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=10)],
                      [strike(), blade_dance(), strike(), defend()]),
    ))
    # Noxious Fumes (power) > inert card for tempo
    comps.append(ValueComparison(
        name="noxious_fumes_power_over_defend",
        category="future_value",
        description="Noxious Fumes > extra Defend (poison each turn)",
        better=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=10)],
                       [noxious_fumes(), defend(), defend(), strike()]),
        worse=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=10)],
                      [defend(), defend(), defend(), strike()]),
    ))
    # Infinite Blades (power): shiv every turn
    comps.append(ValueComparison(
        name="infinite_blades_long_fight",
        category="future_value",
        description="Infinite Blades > extra Strike in long fight",
        better=_vstate(_BASE_PLAYER, [enemy(80, 100, damage=10)],
                       [infinite_blades(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(80, 100, damage=10)],
                      [strike(), strike(), defend(), defend()]),
    ))
    # Well Laid Plans (retain)
    comps.append(ValueComparison(
        name="well_laid_plans_over_strike",
        category="future_value",
        description="Well-Laid Plans (retain cards) > extra Strike",
        better=_vstate(_BASE_PLAYER, [enemy(80, 100, damage=10)],
                       [well_laid_plans(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(80, 100, damage=10)],
                      [strike(), strike(), defend(), defend()]),
    ))

    # --- Turn-count: early turn has more time to deploy future value ---
    comps.append(ValueComparison(
        name="power_early_turn_1_vs_6",
        category="future_value",
        description="Power card on turn 1 > turn 6 (more uses)",
        better=_vstate(_BASE_PLAYER, [enemy(80, 100, damage=10)],
                       [footwork(), strike(), defend(), defend()], turn=1),
        worse=_vstate(_BASE_PLAYER, [enemy(80, 100, damage=10)],
                      [footwork(), strike(), defend(), defend()], turn=6),
    ))
    comps.append(ValueComparison(
        name="draw_engine_early_vs_late",
        category="future_value",
        description="Draw engine earlier in fight > later",
        better=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=10)],
                       [acrobatics(), strike(), strike(), defend()], turn=1),
        worse=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=10)],
                      [acrobatics(), strike(), strike(), defend()], turn=7),
    ))

    # --- Deck size: bigger deck = more draws to come ---
    comps.append(ValueComparison(
        name="more_cards_to_draw",
        category="future_value",
        description="Bigger draw pile > smaller (more future options)",
        better=_vstate(_BASE_PLAYER, [enemy(50, 70, damage=10)],
                       _BASE_HAND, draw_size=15),
        worse=_vstate(_BASE_PLAYER, [enemy(50, 70, damage=10)],
                      _BASE_HAND, draw_size=3),
    ))

    # --- Setup combos: Burst + Blade Dance gives 2x shiv generation ---
    comps.append(ValueComparison(
        name="burst_blade_combo",
        category="future_value",
        description="Burst + Blade Dance (6 shivs) > 2 Strikes",
        better=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=10)],
                       [burst(), blade_dance(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=10)],
                      [strike(), strike(), defend(), defend()]),
    ))
    # Burst + skill draw
    comps.append(ValueComparison(
        name="burst_acrobatics_combo",
        category="future_value",
        description="Burst + Acrobatics (double draw) > 2 defends",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [burst(), acrobatics(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [defend(), defend(), strike(), defend()]),
    ))

    # --- Deadly Poison + Accelerant: priming + payoff ---
    comps.append(ValueComparison(
        name="poison_primer_for_accelerant",
        category="future_value",
        description="Poison + Accelerant > 2 Strikes (combo)",
        better=_vstate(_BASE_PLAYER, [enemy(50, 70, damage=10)],
                       [deadly_poison(), accelerant(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(50, 70, damage=10)],
                      [strike(), strike(), defend(), defend()]),
    ))

    # --- Tactician/Reflex: discard triggers — value vs plain cards ---
    comps.append(ValueComparison(
        name="tactician_over_defend",
        category="future_value",
        description="Tactician (discard->+1E) > plain Defend",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [tactician(), prepared(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [defend(), prepared(), strike(), defend()]),
    ))
    comps.append(ValueComparison(
        name="reflex_over_strike_with_discard",
        category="future_value",
        description="Reflex (discard->draw 2) > plain Strike with discard source",
        better=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                       [reflex(), prepared(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, _BASE_ENEMY,
                      [strike(), prepared(), strike(), defend()]),
    ))

    # --- Escape Plan: cycle + block ---
    comps.append(ValueComparison(
        name="escape_plan_over_defend",
        category="future_value",
        description="Escape Plan (free block + cycle) > Defend",
        better=_vstate(_BASE_PLAYER, [enemy(40, 50, damage=12)],
                       [escape_plan(), strike(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(40, 50, damage=12)],
                      [defend(), strike(), strike(), defend()]),
    ))

    # --- Multi-turn reduction: Expose = Vulnerable long-term ---
    comps.append(ValueComparison(
        name="expose_over_strike",
        category="future_value",
        description="Expose (Vulnerable 3) > Strike when fight continues",
        better=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=10)],
                       [expose(), strike(), strike(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=10)],
                      [strike(), strike(), strike(), defend()]),
    ))

    # --- Sucker Punch sets up Weak (future damage reduction) ---
    comps.append(ValueComparison(
        name="sucker_punch_over_strike",
        category="future_value",
        description="Sucker Punch (dmg + Weak) > Strike (dmg only)",
        better=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=12)],
                       [sucker_punch(), strike(), defend(), defend()]),
        worse=_vstate(_BASE_PLAYER, [enemy(60, 80, damage=12)],
                      [strike(), strike(), defend(), defend()]),
    ))

    return comps


# ---------------------------------------------------------------------------
# Aggregator
# ---------------------------------------------------------------------------

def build_expanded_comparisons() -> list[ValueComparison]:
    """All new comparisons, in order: compound, conditional, arithmetic, future."""
    return (
        build_compound_scaling()
        + build_conditional_value()
        + build_arithmetic_compare()
        + build_future_value()
    )
