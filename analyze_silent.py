#!/usr/bin/env python
"""Analyze 33 Silent run logs from gen6 for Slay the Spire 2."""

import json
import glob
import os
import re
from collections import Counter, defaultdict

LOG_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs", "gen6")


def parse_run(filepath):
    """Parse a single JSONL run log and extract key metrics."""
    events = []
    with open(filepath, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                events.append(json.loads(line))

    run = {
        "file": os.path.basename(filepath),
        "events": events,
    }

    # run_start
    start = events[0]
    run["character"] = start.get("character", "")
    run["run_id"] = start.get("run_id", "")
    run["starting_hp"] = start.get("hp", 0)
    run["starting_max_hp"] = start.get("max_hp", 0)
    run["starting_deck"] = start.get("deck", [])
    run["starting_relics"] = start.get("relics", [])

    # run_end
    end_event = None
    for e in reversed(events):
        if e["type"] == "run_end":
            end_event = e
            break

    if end_event:
        run["outcome"] = end_event.get("outcome", "unknown")
        run["final_floor"] = end_event.get("floor", 0)
        run["final_deck"] = end_event.get("final_deck", [])
        run["final_relics"] = end_event.get("final_relics", [])
        run["final_hp"] = end_event.get("final_hp", 0)
        run["final_max_hp"] = end_event.get("final_max_hp", 0)
        run["final_gold"] = end_event.get("final_gold", 0)
    else:
        run["outcome"] = "incomplete"
        run["final_floor"] = 0
        run["final_deck"] = []
        run["final_relics"] = []
        run["final_hp"] = 0
        run["final_max_hp"] = 0
        run["final_gold"] = 0

    # Combats
    combats = []
    current_combat_floor = None
    current_combat_enemies = []
    combat_starts_for_floor = {}  # track the FIRST combat_start per floor (before retries)

    for e in events:
        if e["type"] == "combat_start":
            floor = e.get("floor", 0)
            enemies = [en["name"] for en in e.get("enemies", [])]
            if floor not in combat_starts_for_floor:
                combat_starts_for_floor[floor] = enemies
            current_combat_floor = floor
            current_combat_enemies = enemies
        elif e["type"] == "combat_end":
            combats.append({
                "floor": current_combat_floor,
                "enemies": combat_starts_for_floor.get(current_combat_floor, current_combat_enemies),
                "outcome": e.get("outcome", ""),
                "turns": e.get("turns", 0),
                "hp_before": e.get("hp_before", 0),
                "hp_after": e.get("hp_after", 0),
                "hp_lost": e.get("hp_before", 0) - e.get("hp_after", 0),
            })
    run["combats"] = combats

    # Find killing enemy (last combat if defeat)
    if run["outcome"] == "defeat" and combats:
        last_combat = combats[-1]
        run["killed_by"] = ", ".join(last_combat["enemies"])
        run["killed_on_floor"] = last_combat["floor"]
    else:
        run["killed_by"] = None
        run["killed_on_floor"] = None

    # Card rewards and picks (deduplicate repeated screens via prompt fingerprint)
    card_picks = []
    card_skips = 0
    seen_reward_prompts = set()
    for e in events:
        if e["type"] == "decision" and e.get("screen_type") == "card_reward":
            choice = e.get("choice", {})
            action = choice.get("action", "")
            prompt = e.get("user_prompt", "")

            # Deduplicate: use first ~200 chars of prompt as fingerprint
            fingerprint = prompt[:200]
            if fingerprint in seen_reward_prompts:
                continue
            seen_reward_prompts.add(fingerprint)

            # Extract offered cards from the prompt
            offered = []
            for m in re.finditer(r"option_index=\d+:\s+(.+?)(?:\s+\()", prompt):
                offered.append(m.group(1).strip())

            if action == "choose_reward_card":
                idx = choice.get("option_index", 0)
                if idx is not None and idx < len(offered):
                    picked = offered[idx]
                else:
                    picked = "unknown"
                card_picks.append({
                    "picked": picked,
                    "offered": offered,
                    "floor": None,
                })
            elif action == "skip_reward_cards":
                card_skips += 1
                card_picks.append({
                    "picked": None,
                    "offered": offered,
                    "floor": None,
                })

    run["card_picks"] = card_picks
    run["card_skips"] = card_skips

    # Potions
    potions_used = []
    potions_gained = []
    potions_discarded = []
    for e in events:
        if e["type"] == "potion_change":
            if e.get("previous") is None and e.get("potion") is not None:
                potions_gained.append(e["potion"])
            elif e.get("potion") is None and e.get("previous") is not None:
                potions_used.append(e["previous"])  # could be used or discarded, hard to tell

    run["potions_gained"] = potions_gained
    run["potions_used"] = potions_used

    # Relics gained
    relics_gained = []
    for e in events:
        if e["type"] == "relic_gained":
            relics_gained.append(e.get("name", e.get("relic_id", "unknown")))
    run["relics_gained"] = relics_gained

    # HP changes through run
    hp_timeline = []
    for e in events:
        if e["type"] == "combat_end":
            hp_timeline.append((e.get("hp_after", 0), "floor_combat"))

    run["hp_timeline"] = hp_timeline

    return run


def get_act(floor):
    """Determine act from floor number."""
    if floor <= 17:
        return 1
    elif floor <= 34:
        return 2
    elif floor <= 52:
        return 3
    else:
        return 4


def print_separator(title):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}")


def main():
    files = sorted(glob.glob(os.path.join(LOG_DIR, "*.jsonl")))

    # Parse only Silent runs
    runs = []
    for f in files:
        run = parse_run(f)
        if run["character"] == "The Silent":
            runs.append(run)

    print(f"SLAY THE SPIRE 2 - SILENT RUN ANALYSIS (gen6)")
    print(f"Total Silent runs analyzed: {len(runs)}")
    print(f"Date range: all from 2026-04-03")

    # =========================================================
    # OVERALL RESULTS
    # =========================================================
    print_separator("OVERALL RESULTS")

    wins = sum(1 for r in runs if r["outcome"] == "win")
    losses = sum(1 for r in runs if r["outcome"] == "defeat")
    print(f"Wins:     {wins}/{len(runs)}  ({100*wins/len(runs):.1f}%)")
    print(f"Losses:   {losses}/{len(runs)}  ({100*losses/len(runs):.1f}%)")

    floors = [r["final_floor"] for r in runs]
    avg_floor = sum(floors) / len(floors) if floors else 0
    print(f"\nAverage floor reached: {avg_floor:.1f}")
    print(f"Median floor reached:  {sorted(floors)[len(floors)//2]}")
    print(f"Min floor:             {min(floors)}")
    print(f"Max floor:             {max(floors)}")

    # =========================================================
    # FLOOR DISTRIBUTION
    # =========================================================
    print_separator("FLOOR DISTRIBUTION (where runs ended)")

    floor_counts = Counter(floors)
    # Group by act
    act_deaths = Counter()
    for f in floors:
        act_deaths[get_act(f)] += 1

    print("\nBy Act:")
    for act in sorted(act_deaths.keys()):
        count = act_deaths[act]
        bar = "#" * count
        print(f"  Act {act}: {count:2d} runs ({100*count/len(runs):5.1f}%)  {bar}")

    print("\nBy Floor:")
    for floor in sorted(floor_counts.keys()):
        count = floor_counts[floor]
        bar = "#" * count
        act = get_act(floor)
        print(f"  Floor {floor:2d} (Act {act}): {count:2d}  {bar}")

    # =========================================================
    # CAUSE OF DEATH
    # =========================================================
    print_separator("CAUSE OF DEATH - WHICH ENEMIES KILL US")

    kill_counts = Counter()
    kill_floors = defaultdict(list)
    for r in runs:
        if r["killed_by"]:
            kill_counts[r["killed_by"]] += 1
            kill_floors[r["killed_by"]].append(r["killed_on_floor"])

    print(f"\n{'Enemy':<45} {'Deaths':>6} {'Avg Floor':>10}")
    print("-" * 65)
    for enemy, count in kill_counts.most_common(20):
        avg_fl = sum(kill_floors[enemy]) / len(kill_floors[enemy])
        bar = "#" * count
        print(f"  {enemy:<43} {count:>4}   floor {avg_fl:4.1f}  {bar}")

    # Boss vs elite vs normal
    print("\nDeath context (boss floors are 17, 34, 52):")
    boss_floors = {17, 34, 52}
    boss_deaths = sum(1 for r in runs if r.get("killed_on_floor") in boss_floors)
    non_boss_deaths = len(runs) - boss_deaths - wins
    print(f"  Died to boss:     {boss_deaths:2d} ({100*boss_deaths/len(runs):.1f}%)")
    print(f"  Died before boss: {non_boss_deaths:2d} ({100*non_boss_deaths/len(runs):.1f}%)")

    # =========================================================
    # COMBAT PERFORMANCE
    # =========================================================
    print_separator("COMBAT PERFORMANCE")

    all_combats = []
    for r in runs:
        for c in r["combats"]:
            all_combats.append(c)

    won_combats = [c for c in all_combats if c["outcome"] == "win"]
    lost_combats = [c for c in all_combats if c["outcome"] == "defeat"]

    print(f"Total combats fought: {len(all_combats)}")
    print(f"Combats won:          {len(won_combats)} ({100*len(won_combats)/len(all_combats):.1f}%)")
    print(f"Combats lost:         {len(lost_combats)} ({100*len(lost_combats)/len(all_combats):.1f}%)")

    if won_combats:
        avg_hp_lost = sum(c["hp_lost"] for c in won_combats) / len(won_combats)
        print(f"\nAvg HP lost per combat won: {avg_hp_lost:.1f}")

    # Most damaging enemies (in combats we won)
    print("\nMost damaging enemies (avg HP lost in wins):")
    enemy_hp_loss = defaultdict(list)
    for c in won_combats:
        key = ", ".join(c["enemies"])
        enemy_hp_loss[key].append(c["hp_lost"])

    enemy_avg = [(name, sum(losses)/len(losses), len(losses))
                  for name, losses in enemy_hp_loss.items() if len(losses) >= 2]
    enemy_avg.sort(key=lambda x: -x[1])

    print(f"  {'Enemy':<45} {'Avg HP Lost':>10} {'Fights':>7}")
    print("  " + "-" * 65)
    for name, avg, count in enemy_avg[:15]:
        print(f"  {name:<45} {avg:>8.1f}   {count:>5}")

    # =========================================================
    # CARD ANALYSIS
    # =========================================================
    print_separator("CARD PICK PATTERNS")

    all_picked = Counter()
    all_offered = Counter()
    all_skipped_when_offered = Counter()

    for r in runs:
        for cp in r["card_picks"]:
            for card in cp["offered"]:
                all_offered[card] += 1
            if cp["picked"]:
                all_picked[cp["picked"]] += 1
            else:
                for card in cp["offered"]:
                    all_skipped_when_offered[card] += 1

    total_rewards = sum(1 for r in runs for cp in r["card_picks"])
    total_skips = sum(r["card_skips"] for r in runs)
    print(f"Total card reward screens: {total_rewards}")
    print(f"Total skips:               {total_skips} ({100*total_skips/total_rewards:.1f}% skip rate)")

    print(f"\nMost picked cards:")
    print(f"  {'Card':<30} {'Picked':>7} {'Offered':>8} {'Pick Rate':>10}")
    print("  " + "-" * 58)
    for card, count in all_picked.most_common(20):
        offered = all_offered.get(card, count)
        rate = 100 * count / offered if offered > 0 else 0
        print(f"  {card:<30} {count:>5}   {offered:>6}    {rate:>6.1f}%")

    print(f"\nMost offered but NEVER picked (potential blind spots):")
    never_picked = [(card, ct) for card, ct in all_offered.most_common()
                     if card not in all_picked and ct >= 3]
    for card, ct in never_picked[:10]:
        print(f"  {card:<30} offered {ct} times, picked 0")

    print(f"\nMost skipped cards (offered but not picked):")
    skip_rates = []
    for card, offered_ct in all_offered.items():
        picked_ct = all_picked.get(card, 0)
        skipped_ct = offered_ct - picked_ct
        if offered_ct >= 3:
            skip_rates.append((card, skipped_ct, offered_ct, 100*skipped_ct/offered_ct))
    skip_rates.sort(key=lambda x: -x[3])
    print(f"  {'Card':<30} {'Skipped':>8} {'Offered':>8} {'Skip Rate':>10}")
    print("  " + "-" * 60)
    for card, skipped, offered, rate in skip_rates[:15]:
        print(f"  {card:<30} {skipped:>6}   {offered:>6}    {rate:>6.1f}%")

    # =========================================================
    # DECK COMPOSITION: WINNING VS LOSING
    # =========================================================
    print_separator("FINAL DECK ANALYSIS")

    # Since all runs lost, analyze by how far we got
    deep_runs = [r for r in runs if r["final_floor"] >= 17]  # reached boss
    early_deaths = [r for r in runs if r["final_floor"] < 10]
    mid_deaths = [r for r in runs if 10 <= r["final_floor"] < 17]

    print(f"Early deaths (floor <10):  {len(early_deaths)} runs")
    print(f"Mid deaths (floor 10-16):  {len(mid_deaths)} runs")
    print(f"Reached Act 1 boss (17+):  {len(deep_runs)} runs")

    def deck_card_freq(run_list, label):
        card_counts = Counter()
        for r in run_list:
            for card in r["final_deck"]:
                # Normalize: strip upgrade markers
                base = card.rstrip("+")
                card_counts[card] += 1
        if not run_list:
            return
        print(f"\n  {label} - avg deck size: {sum(len(r['final_deck']) for r in run_list)/len(run_list):.1f}")
        print(f"  {'Card':<30} {'Appearances':>12} {'% of runs':>10}")
        print("  " + "-" * 55)
        for card, ct in card_counts.most_common(20):
            pct = 100 * ct / len(run_list)
            print(f"  {card:<30} {ct:>10}     {pct:>6.1f}%")

    deck_card_freq(deep_runs, "DEEP RUNS (floor 17+)")
    deck_card_freq(early_deaths, "EARLY DEATHS (floor <10)")

    # Average deck sizes
    all_deck_sizes = [len(r["final_deck"]) for r in runs]
    print(f"\nOverall avg deck size: {sum(all_deck_sizes)/len(all_deck_sizes):.1f}")

    # =========================================================
    # RELIC ANALYSIS
    # =========================================================
    print_separator("RELIC ANALYSIS")

    relic_counts = Counter()
    relic_in_deep = Counter()
    relic_in_early = Counter()

    for r in runs:
        for rel in r["final_relics"]:
            relic_counts[rel] += 1
            if r["final_floor"] >= 17:
                relic_in_deep[rel] += 1
            elif r["final_floor"] < 10:
                relic_in_early[rel] += 1

    print(f"{'Relic':<35} {'Total':>6} {'In Deep':>8} {'In Early':>9}")
    print("-" * 62)
    for relic, ct in relic_counts.most_common(25):
        deep = relic_in_deep.get(relic, 0)
        early = relic_in_early.get(relic, 0)
        print(f"  {relic:<33} {ct:>4}    {deep:>5}     {early:>5}")

    # =========================================================
    # POTION USAGE
    # =========================================================
    print_separator("POTION ANALYSIS")

    all_potions_gained = Counter()
    all_potions_used = Counter()
    for r in runs:
        for p in r["potions_gained"]:
            all_potions_gained[p] += 1
        for p in r["potions_used"]:
            all_potions_used[p] += 1

    total_gained = sum(all_potions_gained.values())
    total_used = sum(all_potions_used.values())
    print(f"Total potions gained: {total_gained}")
    print(f"Total potions consumed/lost: {total_used}")

    print(f"\n{'Potion':<30} {'Gained':>7} {'Used':>6}")
    print("-" * 47)
    for pot, ct in all_potions_gained.most_common(15):
        used = all_potions_used.get(pot, 0)
        print(f"  {pot:<28} {ct:>5}   {used:>4}")

    # =========================================================
    # HP EFFICIENCY
    # =========================================================
    print_separator("HP MANAGEMENT")

    # HP at death
    print(f"Average HP entering final (fatal) combat:")
    fatal_hp = [r["combats"][-1]["hp_before"] for r in runs if r["combats"]]
    if fatal_hp:
        print(f"  Mean: {sum(fatal_hp)/len(fatal_hp):.1f}")
        print(f"  Min:  {min(fatal_hp)}")
        print(f"  Max:  {max(fatal_hp)}")

    # Gold at death
    gold_at_end = [r["final_gold"] for r in runs]
    print(f"\nGold at death:")
    print(f"  Mean: {sum(gold_at_end)/len(gold_at_end):.1f}")
    print(f"  Min:  {min(gold_at_end)}")
    print(f"  Max:  {max(gold_at_end)}")
    print(f"  Runs dying with 100+ gold: {sum(1 for g in gold_at_end if g >= 100)}")
    print(f"    (Unspent gold = missed shop opportunities)")

    # =========================================================
    # COMBAT TURN ANALYSIS
    # =========================================================
    print_separator("COMBAT LENGTH ANALYSIS")

    turn_counts = defaultdict(list)
    for r in runs:
        for c in r["combats"]:
            if c["outcome"] == "win":
                key = ", ".join(c["enemies"])
                turn_counts[key].append(c["turns"])

    print("Longest average combats (potential scaling issues):")
    avg_turns = [(name, sum(t)/len(t), len(t)) for name, t in turn_counts.items() if len(t) >= 2]
    avg_turns.sort(key=lambda x: -x[1])
    print(f"  {'Enemy':<45} {'Avg Turns':>10} {'Fights':>7}")
    print("  " + "-" * 65)
    for name, avg, count in avg_turns[:10]:
        print(f"  {name:<45} {avg:>8.1f}   {count:>5}")

    # =========================================================
    # NEOW BONUS ANALYSIS
    # =========================================================
    print_separator("NEOW BONUS CHOICES")

    neow_choices = Counter()
    for r in runs:
        for e in r["events"]:
            if (e["type"] == "decision" and e.get("screen_type") == "event"
                and "Neow" in e.get("user_prompt", "")):
                choice = e.get("choice", {})
                reasoning = choice.get("reasoning", "")[:80]
                # Extract what was chosen from the prompt
                prompt = e.get("user_prompt", "")
                idx = choice.get("option_index")
                if idx is not None:
                    for m in re.finditer(r"option_index=" + str(idx) + r":\s+(.+?)(?:\n|$)", prompt):
                        neow_choices[m.group(1).strip()[:60]] += 1
                break

    print(f"{'Neow Bonus':<62} {'Count':>6}")
    print("-" * 70)
    for bonus, ct in neow_choices.most_common():
        print(f"  {bonus:<60} {ct:>4}")

    # =========================================================
    # ACTIONABLE INSIGHTS
    # =========================================================
    print_separator("ACTIONABLE INSIGHTS & RECOMMENDATIONS")

    print("""
1. WIN RATE: 0% across 33 runs is critically poor. The bot never wins.

2. FLOOR PROGRESSION:""")
    print(f"   - {100*len(early_deaths)/len(runs):.0f}% of runs die before floor 10 (Act 1 hallway fights)")
    print(f"   - {100*len(deep_runs)/len(runs):.0f}% of runs reach the Act 1 boss (floor 17)")

    if kill_counts:
        top_killer = kill_counts.most_common(1)[0]
        print(f"\n3. TOP KILLER: '{top_killer[0]}' kills us {top_killer[1]} times ({100*top_killer[1]/len(runs):.0f}%)")

    print(f"""
4. ECONOMY: Avg gold at death = {sum(gold_at_end)/len(gold_at_end):.0f}
   - {sum(1 for g in gold_at_end if g >= 100)} runs died with 100+ gold unspent
   - Bot may not be visiting shops or spending gold efficiently

5. DECK BLOAT: Avg deck size = {sum(all_deck_sizes)/len(all_deck_sizes):.1f}
   - Skip rate on card rewards: {100*total_skips/total_rewards:.0f}%
   - Consider skipping more aggressively to keep deck lean""")

    if fatal_hp:
        low_hp_deaths = sum(1 for hp in fatal_hp if hp <= 20)
        print(f"""
6. HP MANAGEMENT: {low_hp_deaths}/{len(runs)} runs entered fatal combat with <=20 HP
   - Bot may be taking too much chip damage in hallway fights
   - Rest more aggressively at campfires when low""")

    # Card archetype analysis
    print("\n7. ARCHETYPE ANALYSIS:")
    shiv_cards = {"Cloak and Dagger", "Accuracy", "Blade Dance", "Infinite Blades",
                  "Storm of Steel", "After Image", "A Thousand Cuts"}
    poison_cards = {"Deadly Poison", "Noxious Fumes", "Crippling Cloud", "Corpse Explosion",
                    "Bouncing Flask", "Catalyst", "Envenom", "Poisoned Stab"}
    discard_cards = {"Prepared", "Acrobatics", "Calculated Gamble", "Tactician",
                     "Reflex", "Concentrate", "Expertise"}

    for r in runs:
        deck_set = set()
        for card in r["final_deck"]:
            deck_set.add(card.rstrip("+"))
        r["has_shiv"] = bool(deck_set & shiv_cards)
        r["has_poison"] = bool(deck_set & poison_cards)
        r["has_discard"] = bool(deck_set & discard_cards)

    shiv_runs = [r for r in runs if r["has_shiv"]]
    poison_runs = [r for r in runs if r["has_poison"]]
    discard_runs = [r for r in runs if r["has_discard"]]

    def avg_floor(run_list):
        if not run_list:
            return 0
        return sum(r["final_floor"] for r in run_list) / len(run_list)

    print(f"   Shiv-focused:    {len(shiv_runs):2d} runs, avg floor {avg_floor(shiv_runs):.1f}")
    print(f"   Poison-focused:  {len(poison_runs):2d} runs, avg floor {avg_floor(poison_runs):.1f}")
    print(f"   Discard-focused: {len(discard_runs):2d} runs, avg floor {avg_floor(discard_runs):.1f}")
    neither = [r for r in runs if not r["has_shiv"] and not r["has_poison"] and not r["has_discard"]]
    print(f"   No archetype:    {len(neither):2d} runs, avg floor {avg_floor(neither):.1f}")

    print("\n8. SPECIFIC RECOMMENDATIONS:")
    print("   - Prioritize BLOCK cards (Leg Sweep, Dodge and Roll, Deflect)")
    print("   - Commit to ONE archetype (poison OR shiv) early")
    print("   - Remove Strikes aggressively at shops/events")
    print("   - Use potions before boss fights, not hallway fights")
    print("   - If entering a boss with <30 HP, consider resting instead of upgrading")
    print("   - Track which elites are safe to fight vs avoid on the map")


if __name__ == "__main__":
    main()
