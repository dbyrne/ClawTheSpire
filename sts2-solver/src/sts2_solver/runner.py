"""Autonomous game runner with static TUI.

Three-panel layout:
  - Top: run status bar (floor, HP, gold, screen)
  - Middle left: Solver output (last combat solution)
  - Middle right: Advisor output (last strategic decision)
  - Bottom: scrolling action log

Usage:
    python run.py                        # auto-play Ironclad from main menu
    python run.py --step                 # step mode: press Enter for each action
    python run.py --dry-run              # show decisions without executing
    python run.py --character Silent     # pick a different character
"""

from __future__ import annotations

import argparse
from collections import deque
import json
import os
import time
from enum import Enum
from pathlib import Path

from rich.console import Console, Group
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from .deterministic_advisor import (
    AUTO_ACTIONS,
    Decision,
    decide_boss_relic,
    decide_card_reward,
    decide_deck_select,
    decide_map,
    decide_rest,
    decide_shop,
    detect_screen_type,
)
from .game_data import strip_markup
from .bridge import state_from_mcp
from .constants import CardType, TargetType
from .data_loader import load_cards
from .game_client import GameClient
from .game_data import load_game_data
from .run_logger import RunLogger
from .state_serializer import combat_state_to_json


DEFAULT_CHARACTER = "Ironclad"
MAX_LOG_LINES = 50

# Actions that are safe to auto-execute without network evaluation.
# These are either choiceless (no option_index needed) or confirmations
# of an already-made choice. Anything NOT in this set that gets auto-picked
# with an option_index is a bug — the network should have been asked.
SAFE_AUTO_ACTIONS = {
    # Choiceless progression
    "proceed",
    "open_chest",
    "open_shop_inventory",
    "close_shop_inventory",
    "collect_rewards_and_proceed",
    "return_to_main_menu",
    # Confirmations of already-made choices
    "confirm_selection",
    "confirm_bundle",
    "confirm_modal",
    # Reward claiming (items, not card selection)
    "claim_reward",
    # Skip actions (when decision was already made)
    "skip_reward_cards",
}


class Runner:
    """Autonomous game runner with static TUI."""

    def __init__(
        self,
        step_mode: bool = False,
        dry_run: bool = False,
        poll_interval: float = 1.0,
        character: str = DEFAULT_CHARACTER,
        logs_dir: str | Path | None = None,
        gen: str | None = None,
    ):
        self.step_mode = step_mode
        self.dry_run = dry_run
        self.review_mode = False  # toggled at runtime with R key
        self.poll_interval = poll_interval
        self.character = character
        self._gen_name = gen

        self.console = Console()
        self.client = GameClient()
        self.card_db = None
        self.game_data = None
        self.logger = RunLogger(logs_dir=Path(logs_dir) if logs_dir else None)

        # Structured event store (SQLite + Supabase + WebSocket)
        from .run_store import RunStore
        from .event_server import get_event_server
        self.store = RunStore(event_server=get_event_server())
        self._store_run_started = False

        self.game_state: dict | None = None
        self.turn_count = 0
        self.action_count = 0
        self._combat_logged = False  # True after combat_start is logged, reset on combat_end
        self._turn_cards_played = None  # Accumulates cards across mid-turn re-entries
        self._turn_targets_chosen = None
        self._turn_start_gs = None

        # BetaOne combat MCTS (loaded in _init_deps)
        self._betaone_onnx_path: str | None = None
        self._betaone_card_vocab_json: str | None = None

        # DeckNet card-reward decisions (loaded in _init_deps)
        self._decknet = None
        self._decknet_card_vocab: dict[str, int] | None = None
        self._checkpoint_name: str | None = None
        self._card_reward_handled = False  # Reset when leaving reward screen
        self._deck_select_stuck = False  # Track stuck deck_select screens
        self._stuck_since: float | None = None  # Timestamp when we got stuck
        self._shop_visited = False  # Prevent re-opening shop after closing
        self._shop_snapshot_logged = False
        self._deck_size_after_skip: int | None = None  # Set after skip_reward_cards
        self._last_floor: int | None = None  # Track floor for shop reset
        self._last_screen_key: tuple[str, str] | None = None  # (screen, screen_type)
        self._screen_repeat_count: int = 0  # Same-screen repeat counter
        self._combat_move_indices: dict[tuple[int, str], int] = {}  # Enemy move cycle tracking

        # Run context for network encoding
        self._current_act_id: str = ""
        self._current_boss_id: str = ""

        # Decision routing counters — tracks how each screen_type was resolved
        self._decision_routing: dict[str, dict[str, int]] = {}  # {screen_type: {source: count}}

        # TUI state
        self._status_text = "[dim]Starting...[/dim]"
        self._solver_text = "[dim]No combat yet[/dim]"
        self._advisor_text = "[dim]No decisions yet[/dim]"
        self._log: deque[str] = deque(maxlen=MAX_LOG_LINES)
        self._live: Live | None = None

    # ------------------------------------------------------------------
    # TUI rendering
    # ------------------------------------------------------------------

    def _build_layout(self) -> Layout:
        layout = Layout()
        layout.split_column(
            Layout(name="status", size=3),
            Layout(name="panels", ratio=2),
            Layout(name="log", ratio=3),
        )
        layout["panels"].split_row(
            Layout(name="solver"),
            Layout(name="advisor"),
        )

        # Status bar
        layout["status"].update(
            Panel(Text.from_markup(self._status_text), title="Run Status", border_style="white")
        )

        # Solver panel
        layout["solver"].update(
            Panel(Text.from_markup(self._solver_text), title="Solver", border_style="red")
        )

        # Advisor panel
        layout["advisor"].update(
            Panel(Text.from_markup(self._advisor_text), title="Advisor", border_style="blue")
        )

        # Log panel — show newest entries first so they don't scroll off
        log_text = "\n".join(reversed(self._log)) if self._log else "[dim]Waiting...[/dim]"
        layout["log"].update(
            Panel(Text.from_markup(log_text), title="Action Log", border_style="green")
        )

        return layout

    def _log_action(self, msg: str) -> None:
        self._log.append(msg)

    def _check_review_toggle(self) -> None:
        """Non-blocking hotkey check: R toggles review mode, S snapshots state."""
        try:
            import msvcrt
            while msvcrt.kbhit():
                ch = msvcrt.getch()
                if ch in (b'r', b'R'):
                    self.review_mode = not self.review_mode
                    state = "ON" if self.review_mode else "OFF"
                    self._log_action(
                        f"[bold cyan]Review mode {state}[/bold cyan]"
                        f" (press R to toggle)"
                    )
                    self._refresh()
                elif ch in (b's', b'S'):
                    self._save_eval_snapshot()
        except ImportError:
            import sys
            import select as _sel
            try:
                while _sel.select([sys.stdin], [], [], 0)[0]:
                    ch = sys.stdin.read(1)
                    if ch in ('r', 'R'):
                        self.review_mode = not self.review_mode
                        state = "ON" if self.review_mode else "OFF"
                        self._log_action(
                            f"[bold cyan]Review mode {state}[/bold cyan]"
                            f" (press R to toggle)"
                        )
                        self._refresh()
                    elif ch in ('s', 'S'):
                        self._save_eval_snapshot()
            except Exception:
                pass

    def _save_eval_snapshot(self) -> None:
        """Dump the current game state + user comment to eval_snapshots/ for later
        translation into an eval scenario. Triggered by pressing 'S' during
        auto-play — manages Live so input() can read the comment cleanly."""
        was_live = self._live is not None
        if was_live:
            self._live.stop()
        try:
            self._save_eval_snapshot_nolive()
        finally:
            if was_live:
                self._live.start()

    def _save_eval_snapshot_nolive(self) -> None:
        """Core snapshot logic. Caller must ensure Live is stopped (or absent).
        Used by the step-mode prompt path where Live is already stopped."""
        import re as _re
        from datetime import datetime

        gs = getattr(self, "game_state", None)
        if not gs:
            self._log_action("[yellow]No game state to snapshot yet[/yellow]")
            return

        try:
            comment = input("Snapshot comment (Enter to skip): ").strip()
        except (EOFError, KeyboardInterrupt):
            comment = ""

        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
        slug = _re.sub(r"[^a-z0-9]+", "_", comment.lower()).strip("_")[:40] or "snap"
        out_dir = Path("eval_snapshots")
        out_dir.mkdir(exist_ok=True)
        path = out_dir / f"{ts}_{slug}.json"

        snap = {
            "timestamp": ts,
            "comment": comment,
            "checkpoint": getattr(self, "_checkpoint_name", None),
            "game_state": gs,
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(snap, f, indent=2, ensure_ascii=False, default=str)
        self._log_action(f"[green]Snapshot saved:[/green] {path}")

    def _review_pause(self, summary: str) -> None:
        """Block until user approves a decision. Only active in review mode.

        Controls: Space = apply, R = resume auto, Q = quit run.
        """
        if not self.review_mode:
            return

        if self._live:
            self._live.stop()

        self.console.print()
        self.console.print("[bold cyan]━━━ REVIEW ━━━[/bold cyan]")
        self.console.print(summary)
        self.console.print(
            "[dim]Space = apply   S = snapshot   R = resume auto   Q = quit[/dim]"
        )

        while True:
            try:
                import msvcrt
                ch = msvcrt.getch()
            except ImportError:
                import sys, tty, termios
                fd = sys.stdin.fileno()
                old = termios.tcgetattr(fd)
                try:
                    tty.setraw(fd)
                    ch = sys.stdin.read(1).encode()
                finally:
                    termios.tcsetattr(fd, termios.TCSADRAIN, old)

            if ch == b' ':
                break
            elif ch in (b's', b'S'):
                # Live is already stopped; snapshot inline then re-print the
                # prompt so the user can still choose Space/R/Q.
                self._save_eval_snapshot_nolive()
                self.console.print(summary)
                self.console.print(
                    "[dim]Space = apply   S = snapshot   R = resume auto   Q = quit[/dim]"
                )
            elif ch in (b'r', b'R'):
                self.review_mode = False
                self.console.print("[cyan]Review mode OFF — resuming auto[/cyan]")
                break
            elif ch in (b'q', b'Q'):
                self.review_mode = False
                if self._live:
                    self._live.start()
                raise KeyboardInterrupt

        if self._live:
            self._live.start()

    def _review_combat_decision(self, label, sim_state, gs, hand, policy,
                                root_value, solve_ms, child_values=None,
                                child_visits=None) -> None:
        """Consolidated review display for all combat MCTS decisions."""
        if not self.review_mode:
            return
        head_vals = ""
        alts = self._review_combat_alternatives(
            sim_state, hand, policy, child_values, child_visits)
        total_vis = sum(child_visits) if child_visits else 0
        best_vis = max(child_visits) if child_visits else 0
        confidence = best_vis / max(1, total_vis)
        enemy_summary = ", ".join(
            f"{e.name} {e.hp}hp" for e in sim_state.enemies if e.is_alive
        )
        player = gs.get("run") or gs.get("player") or {}
        self._review_pause(
            f"[bold]{label}[/bold]  (MCTS {root_value:+.2f} | "
            f"{confidence:.0%} | {solve_ms:.0f}ms)\n"
            f"{head_vals}\n"
            f"{alts}\n"
            f"  Hand: {', '.join(f'{c.name}({c.cost})' for c in hand)}\n"
            f"  HP {sim_state.player.hp}/{sim_state.player.max_hp} | "
            f"Energy {sim_state.player.energy} | Block {sim_state.player.block}\n"
            f"  vs: {enemy_summary}"
        )

    def _review_combat_alternatives(self, sim_state, hand, policy,
                                     child_values=None, child_visits=None) -> str:
        """Build labeled alternatives from MCTS policy for review display."""
        if not policy or len(policy) < 2:
            return ""
        # Reconstruct action labels matching Rust enumerate_actions order:
        # 1. Playable cards (deduplicated by id+upgraded), with targets
        #    — must skip unplayable cards (cost > energy) to stay aligned
        # 2. Potions
        # 3. End turn
        labels = []
        seen = set()
        alive_enemies = [e for e in sim_state.enemies if e.is_alive]
        energy = sim_state.player.energy
        for i, card in enumerate(hand):
            # Skip unplayable cards — Rust enumerate_actions excludes them
            if card.cost < 0:
                continue  # Status/Curse
            cost = card.cost
            if card.card_type == CardType.SKILL and sim_state.player.powers.get("Corruption", 0) > 0:
                cost = 0
            if cost > energy:
                continue
            key = (card.id, card.upgraded)
            if key in seen:
                continue
            seen.add(key)
            name = f"{card.name}+" if card.upgraded else card.name
            if card.target in (TargetType.ANY_ENEMY,):
                for ti, e in enumerate(alive_enemies):
                    labels.append(f"{name}\u2192{e.name}")
            else:
                labels.append(name)
        for i, pot in enumerate(sim_state.player.potions):
            if pot and (pot.get("name") or pot.get("id")):
                labels.append(f"Potion:{pot.get('name') or pot.get('id', i)}")
        labels.append("End Turn")

        # Pair labels with policy/value/visit data, sort by visits
        entries = []
        for i in range(min(len(labels), len(policy))):
            val = child_values[i] if child_values and i < len(child_values) else None
            vis = child_visits[i] if child_visits and i < len(child_visits) else None
            entries.append((labels[i], policy[i], val, vis))
        entries.sort(key=lambda x: -(x[3] or 0))

        lines = []
        for name, p, val, vis in entries:
            parts = [name]
            if val is not None:
                parts.append(f"val={val:+.2f}")
            if vis is not None:
                parts.append(f"n={vis}")
            lines.append("  " + "  ".join(parts))
        return "\n".join(lines)

    def _track_decision(self, screen_type: str, source: str) -> None:
        """Track decision routing for end-of-run summary."""
        by_source = self._decision_routing.setdefault(screen_type, {})
        by_source[source] = by_source.get(source, 0) + 1

    def _emit_routing_summary(self, outcome: str, floor) -> None:
        """Log decision routing summary to JSONL and TUI at end of run.

        Shows how each decision type was routed (network vs deterministic
        vs auto vs advisor) so we can immediately spot when the network
        is being bypassed for decisions it should be handling.
        """
        # Also scan JSONL for advisor decisions (logged by advisor, not runner)
        routing = {}
        for st, sources in self._decision_routing.items():
            routing[st] = dict(sources)

        log_path = getattr(self.logger, '_path', None)
        if log_path and log_path.exists():
            import json as _json
            try:
                with open(log_path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            ev = _json.loads(line)
                        except Exception:
                            continue
                        if ev.get("type") == "decision" and ev.get("source") == "advisor":
                            st = ev.get("screen_type", "unknown")
                            by_source = routing.setdefault(st, {})
                            by_source["advisor"] = by_source.get("advisor", 0) + 1
            except Exception:
                pass

        if not routing:
            return

        # DeckNet is expected to handle card rewards when a checkpoint is loaded
        expect_decknet = self._decknet is not None

        lines = []
        warnings = []
        for st in sorted(routing):
            parts = [f"{src}={n}" for src, n in sorted(routing[st].items())]
            lines.append(f"  {st}: {', '.join(parts)}")
            if st == "card_reward" and expect_decknet:
                det = sum(n for src, n in routing[st].items() if src != "decknet")
                dn = routing[st].get("decknet", 0)
                if det > 0 and dn == 0:
                    warnings.append(f"  WARN {st}: ALL decisions bypassed DeckNet ({det}x deterministic)")
                elif det > 0:
                    warnings.append(f"  WARN {st}: {det}/{det + dn} bypassed DeckNet")

        # Log to TUI
        self._log_action("[bold]Decision routing:[/bold]")
        for line in lines:
            self._log_action(f"[dim]{line}[/dim]")
        for w in warnings:
            self._log_action(f"[yellow]{w}[/yellow]")

        # Log to JSONL
        self.logger._emit({
            "type": "decision_routing",
            "outcome": outcome,
            "floor": floor,
            "routing": routing,
            "warnings": [w.strip().lstrip("⚠ ") for w in warnings],
        })

        # Reset for next run
        self._decision_routing = {}

    def _update_status(self) -> None:
        gs = self.game_state
        if gs is None:
            self._status_text = "[yellow]Waiting for game...[/yellow]"
            return

        screen = gs.get("screen", "?")
        run = gs.get("run") or {}
        floor = run.get("floor", "?")
        hp = run.get("current_hp", "?")
        max_hp = run.get("max_hp", "?")
        gold = run.get("gold", "?")

        mode = ""
        if self.dry_run:
            mode = " [yellow]\\[DRY RUN][/yellow]"
        elif self.step_mode:
            mode = " [cyan]\\[STEP][/cyan]"
        if self.review_mode:
            mode += " [bold cyan]\\[REVIEW][/bold cyan]"

        parts = [
            f"[bold]{self.character}[/bold]",
            f"Floor {floor}",
            f"HP {hp}/{max_hp}",
            f"Gold {gold}",
            f"Screen: {screen}",
            f"Turns: {self.turn_count}",
            f"Actions: {self.action_count}",
        ]
        self._status_text = " | ".join(parts) + mode

    def _refresh(self) -> None:
        if self._live:
            self._update_status()
            self._live.update(self._build_layout())

    @staticmethod
    def _is_card_reward_item(item: dict) -> bool:
        """Check if a reward item is a card reward (works with both raw and agent_view)."""
        # Raw state: reward_type = "Card"
        rtype = str(item.get("reward_type", "")).lower()
        if rtype == "card":
            return True
        # Agent view: line = "card: Add a card..."
        line = str(item.get("line", "")).lower()
        if line.startswith("card"):
            return True
        return False

    @staticmethod
    def _is_potion_reward_item(item: dict) -> bool:
        """Check if a reward item is a potion reward."""
        rtype = str(item.get("reward_type", "")).lower()
        if rtype == "potion":
            return True
        line = str(item.get("line", "")).lower()
        if "potion" in line:
            return True
        return False

    @staticmethod
    def _potions_full(gs: dict) -> bool:
        """Return True if all 3 potion slots are occupied."""
        run = gs.get("run") or {}
        potions = run.get("potions") or []
        occupied = sum(1 for p in potions if isinstance(p, dict) and p.get("occupied"))
        return occupied >= 3

    # ------------------------------------------------------------------
    # Init & main loop
    # ------------------------------------------------------------------

    def _init_deps(self) -> None:
        from pathlib import Path as _Path
        import json as _json
        import torch

        self.console.print("[dim]Loading card database...[/dim]")
        self.card_db = load_cards()
        self.console.print(f"[dim]Loaded {len(self.card_db)} cards[/dim]")
        self.console.print("[dim]Loading game data...[/dim]")
        self.game_data = load_game_data()

        # Load BetaOne combat checkpoint (required for combat MCTS)
        betaone_ckpt_dir = _Path(__file__).resolve().parents[3] / "betaone_checkpoints"
        betaone_latest = betaone_ckpt_dir / "betaone_latest.pt"
        betaone_vocab = betaone_ckpt_dir / "card_vocab.json"
        if not (betaone_latest.exists() and betaone_vocab.exists()):
            raise RuntimeError(
                f"BetaOne checkpoint missing: expected {betaone_latest} + "
                f"{betaone_vocab}. Train or copy a BetaOne experiment into "
                f"betaone_checkpoints/ before running."
            )
        from .betaone.network import (
            BetaOneNetwork, export_onnx as betaone_export_onnx,
            network_kwargs_from_meta,
        )
        with open(betaone_vocab) as f:
            card_vocab = _json.load(f)
        self._betaone_card_vocab_json = _json.dumps(card_vocab)
        betaone_ckpt = torch.load(betaone_latest, map_location="cpu", weights_only=False)
        betaone_net = BetaOneNetwork(
            num_cards=len(card_vocab),
            **network_kwargs_from_meta(betaone_ckpt.get("arch_meta")),
        )
        betaone_net.load_state_dict(betaone_ckpt["model_state_dict"])
        self._betaone_onnx_path = betaone_export_onnx(betaone_net, str(betaone_ckpt_dir / "onnx"))

        # Resolve the frontier experiment name from FRONTIER.md at repo root,
        # so the runner logs "reanalyse-v3 gen 88" instead of bare "gen 88".
        # Falls back to just-the-gen if FRONTIER.md is missing or its claimed
        # gen doesn't match the actual checkpoint (stale pointer).
        gen = betaone_ckpt.get("gen", "?")
        frontier_md = _Path(__file__).resolve().parents[3] / "FRONTIER.md"
        exp_name = None
        if frontier_md.exists():
            claimed_exp = None
            claimed_gen = None
            for line in frontier_md.read_text(encoding="utf-8").splitlines():
                if line.startswith("---") and claimed_exp is not None:
                    break
                if line.startswith("experiment:"):
                    claimed_exp = line.split(":", 1)[1].strip()
                elif line.startswith("gen:"):
                    try:
                        claimed_gen = int(line.split(":", 1)[1].strip())
                    except ValueError:
                        pass
            if claimed_exp and claimed_gen == gen:
                exp_name = claimed_exp
            elif claimed_exp:
                self.console.print(
                    f"[yellow]FRONTIER.md claims {claimed_exp} gen {claimed_gen} "
                    f"but checkpoint is gen {gen}; run promote_to_frontier.py to sync[/yellow]"
                )
        label = f"{exp_name} gen {gen}" if exp_name else f"gen {gen}"
        self._checkpoint_name = f"betaone {label}"
        self.console.print(f"[green]BetaOne combat MCTS active ({label})[/green]")

        # Load DeckNet card-reward checkpoint (optional — falls back to
        # deterministic card picks if missing)
        decknet_ckpt_dir = _Path(__file__).resolve().parents[3] / "decknet_checkpoints"
        decknet_latest = decknet_ckpt_dir / "decknet_latest.pt"
        decknet_vocab = decknet_ckpt_dir / "card_vocab.json"
        if decknet_latest.exists() and decknet_vocab.exists():
            from .decknet.network import DeckNet
            with open(decknet_vocab) as f:
                dn_vocab = _json.load(f)
            dn_net = DeckNet(num_cards=len(dn_vocab))
            dn_ckpt = torch.load(decknet_latest, map_location="cpu", weights_only=False)
            dn_net.load_state_dict(dn_ckpt["model_state_dict"])
            dn_net.eval()
            self._decknet = dn_net
            self._decknet_card_vocab = dn_vocab
            self.console.print(f"[green]DeckNet card-reward active (gen {dn_ckpt.get('gen', '?')})[/green]")
        else:
            self.console.print("[yellow]No DeckNet checkpoint — card rewards use deterministic advisor[/yellow]")

        self.logger.metadata = {
            "checkpoint": self._checkpoint_name,
            "decknet": self._decknet is not None,
        }
        try:
            health = self.client.get_health()
            self.logger.game_version = health.get("game_version")
            self.console.print(f"[green]Connected to game v{self.logger.game_version}[/green]")
        except ConnectionError:
            self.console.print("[yellow]Game not reachable yet — will retry[/yellow]")

    def _rust_mcts_search(self, sim_state, num_simulations=100, temperature=0):
        """Run BetaOne MCTS via Rust engine. Returns (Action, policy, root_value)."""
        import sts2_engine

        state_json = combat_state_to_json(sim_state)
        result = sts2_engine.betaone_mcts_search(
            state_json=state_json,
            onnx_path=self._betaone_onnx_path,
            card_vocab_json=self._betaone_card_vocab_json,
            num_sims=num_simulations,
            temperature=float(temperature),
            seed=int(time.time() * 1000) & 0xFFFFFFFF,
        )
        return self._parse_mcts_result(result)

    @staticmethod
    def _parse_mcts_result(result):
        from .actions import Action
        action_type = result["action_type"]
        if action_type == "end_turn":
            action = Action(action_type="end_turn")
        elif action_type == "play_card":
            action = Action(action_type="play_card",
                            card_idx=result["card_idx"],
                            target_idx=result.get("target_idx"))
        elif action_type == "use_potion":
            action = Action(action_type="use_potion",
                            potion_idx=result["potion_idx"])
        elif action_type == "choose_card":
            action = Action(action_type="choose_card",
                            choice_idx=result["choice_idx"])
        else:
            action = Action(action_type="end_turn")

        policy = list(result["policy"])
        root_value = float(result["root_value"])
        child_values = list(result.get("child_values", []))
        child_visits = list(result.get("child_visits", []))
        return action, policy, root_value, child_values, child_visits

    def run(self) -> None:
        self._init_deps()
        finished = False

        with Live(self._build_layout(), console=self.console, refresh_per_second=2, screen=True) as live:
            self._live = live
            try:
                while not finished:
                    finished = self._tick()
                    self._refresh()
                    if finished:
                        break
                    self._check_review_toggle()
                    if self.step_mode:
                        # Drop out of Live temporarily for input
                        live.stop()
                        resp = input("[step] Enter=next, s=snapshot, q=quit: ").strip().lower()
                        if resp == "s":
                            # Live is already stopped; snapshot takes its own
                            # input. Re-start Live and re-prompt rather than
                            # advancing — the user presumably wants to keep
                            # inspecting this state.
                            self._save_eval_snapshot_nolive()
                            live.start()
                            continue
                        if resp == "q":
                            break
                        live.start()
                    else:
                        time.sleep(self.poll_interval)
            except KeyboardInterrupt:
                self._log_action("[yellow]Stopped by user[/yellow]")
                self._refresh()
            finally:
                self._live = None
                self.logger.close()

    # ------------------------------------------------------------------
    # Main tick
    # ------------------------------------------------------------------

    def _tick(self) -> bool:
        """Single iteration. Returns True if run is finished."""
        try:
            self.game_state = self.client.get_state()
        except ConnectionError:
            self._status_text = "[yellow]Waiting for game...[/yellow]"
            return False

        screen = self.game_state.get("screen", "")
        actions = self.game_state.get("available_actions", [])

        # Runtime guard: detect if a skipped card reward was claimed anyway
        if self._deck_size_after_skip is not None:
            current_deck = (self.game_state.get("run") or {}).get("deck", [])
            if len(current_deck) > self._deck_size_after_skip:
                added = len(current_deck) - self._deck_size_after_skip
                self._log_action(
                    f"[bold red]FATAL: Deck grew by {added} after skip_reward_cards! "
                    f"Network decision was overridden.[/bold red]"
                )
                if self.logger:
                    self.logger._emit({
                        "type": "skip_override_detected",
                        "deck_before": self._deck_size_after_skip,
                        "deck_after": len(current_deck),
                    })
                # Don't crash — this decision is already corrupted but the
                # rest of the run's data is still valuable.
                self._deck_size_after_skip = None
            # Clear once we've moved past the reward screen
            if "REWARD" not in screen.upper():
                self._deck_size_after_skip = None

        if screen == "GAME_OVER":
            self._handle_game_over()
            return True

        if screen == "MAIN_MENU":
            self._handle_main_menu(actions)
            return False

        if screen == "CHARACTER_SELECT":
            self._handle_character_select(actions)
            return False

        if screen == "MODAL":
            self._handle_modal(actions)
            return False

        if screen == "CAPSTONE_SELECTION" and "choose_capstone_option" in actions:
            self._handle_capstone_selection(actions)
            return False

        if screen == "BUNDLE_SELECTION" and "confirm_bundle" in actions:
            self._log_action("[dim]auto: confirm_bundle[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("confirm_bundle")
                    time.sleep(1.0)
                except Exception:
                    pass
            return False

        if not actions:
            return False

        self.logger.ensure_run(self.game_state)

        # Start a store run if we haven't yet for this run_id
        if not self._store_run_started:
            run = self.game_state.get("run") or {}
            run_id = self.game_state.get("run_id") or run.get("run_id")
            if run_id:
                self.store.start_run(
                    run_id=run_id,
                    character=run.get("character", self.character),
                    checkpoint=getattr(self, "_checkpoint_name", None),
                    gen=getattr(self, "_gen_name", None),
                    hp=run.get("current_hp"),
                    max_hp=run.get("max_hp"),
                )
                self._store_run_started = True
                self._store_run_id = run_id

                # Detect act and boss from map data
                self._detect_run_context(self.game_state)

        screen = self.game_state.get("screen", "")

        # Reset card reward tracking when we leave the reward screen
        if screen not in ("REWARD", "CARD_SELECTION"):
            self._card_reward_handled = False

        # Reset shop visit flag when the floor changes (not when screen changes)
        run = self.game_state.get("run") or {}
        current_floor = run.get("floor")
        if current_floor is not None and current_floor != self._last_floor:
            self._shop_visited = False
            self._shop_snapshot_logged = False
            self._last_floor = current_floor

        # Reset deck_select stuck flag when we leave the card selection screen
        if screen != "CARD_SELECTION":
            self._deck_select_stuck = False
            self._stuck_since = None

        # If stuck on a screen for too long, force end the run
        if self._deck_select_stuck and self._stuck_since:
            stuck_duration = time.monotonic() - self._stuck_since
            if stuck_duration > 60:
                self._log_action("[red]Stuck for >60s on deck select — forcing run end[/red]")
                self.logger.log_run_end(self.game_state, "stuck")
                if self._store_run_started:
                    run = self.game_state.get("run") or {}
                    self.store.end_run(
                        self._store_run_id, outcome="stuck",
                        floor=run.get("floor", 0),
                        hp=run.get("current_hp"), max_hp=run.get("max_hp"),
                    )
                    self._store_run_started = False
                return True  # Signal run is finished

        in_combat = (
            "play_card" in actions
            or ("end_turn" in actions and "COMBAT" in screen.upper())
        )

        # Track same-screen repeats to detect stuck loops
        screen_type = detect_screen_type(actions) if not in_combat else "combat"
        screen_key = (screen, screen_type)
        if screen_key == self._last_screen_key:
            self._screen_repeat_count += 1
        else:
            self._last_screen_key = screen_key
            self._screen_repeat_count = 0

        # If stuck on the same screen for too many ticks, force a default action.
        # Only safe escapes are allowed — anything else is a bug.
        if self._screen_repeat_count > 5 and not in_combat:
            self._screen_repeat_count = 0  # Reset to avoid infinite force loops
            if screen_type == "shop" and "close_shop_inventory" in actions:
                self._log_action(
                    f"[yellow]Stuck on shop for {self._screen_repeat_count} ticks — closing[/yellow]"
                )
                if not self.dry_run:
                    try:
                        self._execute_with_retry("close_shop_inventory")
                        self._shop_visited = True
                        self.action_count += 1
                    except Exception as e:
                        self._log_action(f"  [red]Forced action failed: {e}[/red]")
            else:
                raise RuntimeError(
                    f"Stuck on {screen}/{screen_type} for >5 ticks. "
                    f"Actions: {actions}. This likely means a screen handler is "
                    f"missing or broken — investigate instead of blindly forcing."
                )
            return False

        if in_combat:
            self._handle_combat()
        else:
            self._handle_non_combat(actions)

        return False

    # ------------------------------------------------------------------
    # Menu / character select
    # ------------------------------------------------------------------

    def _handle_main_menu(self, actions: list[str]) -> None:
        if "abandon_run" in actions:
            self._log_action("[yellow]Abandoning existing run...[/yellow]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("abandon_run")
                    time.sleep(1.0)
                    gs = self.client.get_state()
                    if "confirm_modal" in gs.get("available_actions", []):
                        self._execute_with_retry("confirm_modal")
                        time.sleep(1.0)
                except Exception as e:
                    self._log_action(f"[red]Failed to abandon run: {e}[/red]")
            return

        if "open_character_select" in actions:
            self._log_action("[cyan]Opening character select...[/cyan]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("open_character_select")
                    time.sleep(0.5)
                except Exception as e:
                    self._log_action(f"[red]Failed: {e}[/red]")

    def _handle_character_select(self, actions: list[str]) -> None:
        gs = self.game_state
        char_select = gs.get("character_select") or {}
        characters = char_select.get("characters") or []
        selected = char_select.get("selected_character_id")
        can_embark = char_select.get("can_embark", False)

        target_idx = None
        target_name = None
        for char in characters:
            name = char.get("name", "")
            char_id = char.get("character_id", "")
            if (self.character.lower() in name.lower()
                    or self.character.lower() in char_id.lower()):
                if not char.get("is_locked", False):
                    target_idx = char.get("index")
                    target_name = name
                    break

        if target_idx is None:
            available = [
                c.get("name", c.get("character_id", "?"))
                for c in characters if not c.get("is_locked", False)
            ]
            self._log_action(
                f"[red]Character '{self.character}' not found. "
                f"Available: {', '.join(available)}[/red]"
            )
            return

        needs_select = selected is None or self.character.lower() not in (selected or "").lower()
        if needs_select and "select_character" in actions:
            self._log_action(f"[cyan]Selecting {target_name}...[/cyan]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("select_character", option_index=target_idx)
                    time.sleep(0.5)
                except Exception as e:
                    self._log_action(f"[red]Failed to select: {e}[/red]")
            return

        if can_embark and "embark" in actions:
            self._log_action(f"[bold cyan]Embarking as {target_name}![/bold cyan]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("embark")
                    time.sleep(2.0)
                except Exception as e:
                    self._log_action(f"[red]Failed to embark: {e}[/red]")

    # ------------------------------------------------------------------
    # Modals
    # ------------------------------------------------------------------

    def _handle_modal(self, actions: list[str]) -> None:
        gs = self.game_state
        modal = gs.get("modal") or {}
        modal_type = modal.get("type", "")
        confirm_label = (modal.get("confirm_label") or "").lower()

        is_tutorial = any(
            kw in modal_type.lower()
            for kw in ("tutorial", "hint", "tip", "help", "learn")
        ) or any(
            kw in confirm_label
            for kw in ("tutorial", "yes", "learn", "sure")
        )

        if is_tutorial and "dismiss_modal" in actions:
            self._log_action(f"[dim]Dismissed tutorial: {modal_type}[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("dismiss_modal")
                    time.sleep(0.5)
                except Exception:
                    pass
            return

        if "dismiss_modal" in actions:
            self._log_action(f"[dim]Dismissed modal: {modal_type}[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("dismiss_modal")
                    time.sleep(0.5)
                except Exception:
                    pass
        elif "confirm_modal" in actions:
            self._log_action(f"[dim]Confirmed modal: {modal_type}[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("confirm_modal")
                    time.sleep(0.5)
                except Exception:
                    pass

    def _handle_capstone_selection(self, actions: list[str]) -> None:
        """Handle capstone/relic pack selection screens."""
        import json as _json
        gs = self.game_state
        self._log_action(
            "[red]UNHANDLED AUTO: choose_capstone_option — game state dumped below[/red]"
        )
        for key in ("selection", "chest", "reward", "agent_view"):
            val = gs.get(key)
            if val:
                self._log_action(f"  [yellow]gs[{key}]: {_json.dumps(val)[:500]}[/yellow]")
        raise RuntimeError(
            "choose_capstone_option requires network evaluation but was about to be "
            "auto-picked. Investigate game state to wire up network decision."
        )

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    # Potion categories for smart usage decisions
    _POTION_CATS: dict[str, set[str]] = {
        "heal":     {"heal", "blood", "fairy", "fruit", "regen"},
        "block":    {"block", "ghost", "shield", "iron", "armor"},
        "damage":   {"fire", "attack", "explosive", "damage", "poison",
                     "lightning", "bomb", "swift"},
        "buff":     {"strength", "flex", "dexterity", "energy", "speed",
                     "power", "stance"},
        "debuff":   {"vulnerable", "weak", "fear"},
    }

    def _classify_potion(self, name: str, desc: str) -> str | None:
        """Return the category of a potion, or None if unrecognized."""
        text = f"{name} {desc}".lower()
        for cat, keywords in self._POTION_CATS.items():
            if any(kw in text for kw in keywords):
                return cat
        return None

    def _best_damage_target(self, enemies: list[dict]) -> int | None:
        """Pick the alive enemy with the lowest HP (most likely to kill)."""
        best_idx, best_hp = None, float("inf")
        for e in enemies:
            if e.get("current_hp", 0) <= 0:
                continue
            if e.get("current_hp", 0) < best_hp:
                best_hp = e["current_hp"]
                best_idx = e.get("index", 0)
        return best_idx

    def _should_use_potion(self, gs: dict) -> tuple[int, int | None] | None:
        """Decide whether to use a potion this turn. Returns (slot, target) or None.

        Strategy — save potions for boss fights:
        - Survival (any fight): use block/heal if we'd die, heal if HP < 35%
        - Non-boss fights: save all potions after survival checks
        - Boss fights: use potions aggressively (buff/debuff early, damage any time)
        """
        if "use_potion" not in gs.get("available_actions", []):
            return None

        run = gs.get("run") or {}
        potions = run.get("potions", [])
        combat = gs.get("combat") or {}
        player = combat.get("player") or {}
        enemies = combat.get("enemies") or []

        hp = player.get("current_hp", 0)
        max_hp = player.get("max_hp", 1)
        block = player.get("block", 0)
        turn = gs.get("turn", 0)

        # Calculate total incoming damage
        total_incoming = 0
        for e in enemies:
            if e.get("current_hp", 0) <= 0:
                continue
            for intent in e.get("intents", []):
                if intent.get("intent_type") == "Attack":
                    dmg = intent.get("damage", 0) * intent.get("hits", 1)
                    total_incoming += dmg

        unblocked = max(0, total_incoming - block)
        would_die = unblocked >= hp
        hp_pct = hp / max_hp if max_hp > 0 else 1.0

        alive_enemies = [e for e in enemies if e.get("current_hp", 0) > 0]
        total_enemy_hp = sum(e.get("current_hp", 0) for e in alive_enemies)

        # Detect boss fights via floor number (most reliable)
        from .config import STRATEGY
        floor = run.get("floor", 0)
        is_boss = floor in STRATEGY.get("boss_floors", set())

        # Fallback: very high enemy HP likely means boss
        if not is_boss and any(e.get("max_hp", 0) > 120 for e in alive_enemies):
            is_boss = True

        # Collect usable potions by category
        usable: list[tuple[int, str | None, bool]] = []  # (slot, cat, needs_target)
        for pot in potions:
            if not pot.get("occupied") or not pot.get("can_use"):
                continue
            slot = pot.get("index", 0)
            name = (pot.get("name") or "")
            desc = (pot.get("description") or "")
            cat = self._classify_potion(name, desc)
            needs_target = pot.get("requires_target", False)
            usable.append((slot, cat, needs_target))

        if not usable:
            return None

        first_alive = self._best_damage_target(enemies)

        def _target(needs_target: bool) -> int | None:
            return first_alive if needs_target else None

        # --- Priority 1: Survival — use block/heal potions if we'd die ---
        if would_die:
            for slot, cat, needs_target in usable:
                if cat == "block":
                    return (slot, _target(needs_target))
            for slot, cat, needs_target in usable:
                if cat == "heal":
                    return (slot, _target(needs_target))
            # Offense as defense — kill them before they kill us
            for slot, cat, needs_target in usable:
                if cat == "damage":
                    return (slot, _target(needs_target))

        # --- Priority 2: Heal if HP is critical ---
        if hp_pct < 0.35:
            for slot, cat, needs_target in usable:
                if cat == "heal":
                    return (slot, _target(needs_target))

        # --- Non-boss fights: save potions for the boss ---
        if not is_boss:
            return None

        # --- Boss fight: use potions aggressively ---
        # Buff/debuff potions on early turns for max value
        if turn <= 2:
            for slot, cat, needs_target in usable:
                if cat in ("buff", "debuff"):
                    return (slot, _target(needs_target))
        # Damage potions any time during boss fights
        for slot, cat, needs_target in usable:
            if cat == "damage":
                return (slot, _target(needs_target))
        # Buff potions are still good mid-fight
        for slot, cat, needs_target in usable:
            if cat == "buff":
                return (slot, _target(needs_target))
        # Debuff potions too
        for slot, cat, needs_target in usable:
            if cat == "debuff":
                return (slot, _target(needs_target))

        return None

    def _handle_combat(self) -> None:
        gs = self.game_state
        combat = gs.get("combat") or {}
        player = combat.get("player") or {}
        enemies = combat.get("enemies") or []
        turn = gs.get("turn", "?")

        alive = [e for e in enemies if e.get("current_hp", 0) > 0]
        enemy_str = ", ".join(
            f"{e.get('name', '?')} {e.get('current_hp', '?')}hp" for e in alive
        )
        self._log_action(
            f"[red]Combat T{turn}[/red] | vs {enemy_str}"
        )

        if not self._combat_logged:
            # First entry into this combat — log combat_start once.
            # After deck_select screens (Survivor discard), the game may
            # re-enter _handle_combat with turn=1, but we don't re-log.
            time.sleep(0.5)
            gs = self.client.get_state()
            self.game_state = gs
            combat = gs.get("combat") or {}
            player = combat.get("player") or {}
            enemies = combat.get("enemies") or []

            self.logger.log_combat_start(gs)
            self._combat_logged = True
            self._combat_move_indices = {}
            self.turn_count = 0

            # Snapshot encounter for recording
            run = gs.get("run") or {}
            hand = (combat.get("player") or {}).get("hand") or []
            self._combat_encounter = {
                "enemy_ids": [e.get("enemy_id", e.get("id", e.get("name", "?"))) for e in enemies],
                "enemy_names": [e.get("name", "?") for e in enemies],
                "enemy_hps": [e.get("max_hp", 0) for e in enemies],
                "deck": [c.get("card_id", c.get("id", "?")) for c in (run.get("deck") or [])],
                "player_hp": player.get("current_hp", 0),
                "player_max_hp": player.get("max_hp", 70),
                "floor": run.get("floor", 0),
                "act": run.get("act_id", ""),
                "relics": [r.get("relic_id", r.get("id", r.get("name", "?"))) for r in (run.get("relics") or [])],
            }
            self._combat_start_hp = player.get("current_hp", 0)

            if self._store_run_started:
                run = gs.get("run") or {}
                self.store.log_combat_start(
                    self._store_run_id, floor=run.get("floor", 0),
                    hp=player.get("current_hp", 0), max_hp=player.get("max_hp", 0),
                    enemies=[e.get("name", "?") for e in enemies],
                )

        # Potions are now handled by MCTS as part of the action space —
        # the network decides when to use potions during the play loop.

        # turn_start_gs is set on first entry (see accumulation block below)

        # MCTS with hybrid evaluation: network policy priors guide the
        # search, hand-crafted evaluator scores within-turn states for
        # correct damage/block tradeoffs, network value head adds
        # multi-turn strategic context.
        from .bridge import action_to_mcp

        # Accumulate across re-entries (discard prompts break the play loop
        # and re-enter _handle_combat; we merge cards from the same turn).
        if self._turn_cards_played is None:
            self._turn_cards_played = []
            self._turn_targets_chosen = []
            self._turn_start_gs = gs
            self._potions_used_this_turn = set()
        cards_played = self._turn_cards_played
        targets_chosen = self._turn_targets_chosen
        turn_start_gs = self._turn_start_gs
        total_states = 0
        total_solve_ms = 0.0
        best_score = 0.0
        turn_root_value: float | None = None
        max_cards = 12

        while len(cards_played) < max_cards:
            self._check_review_toggle()
            # Build combat state and run MCTS
            try:
                sim_state = state_from_mcp(gs, self.card_db,
                                          move_indices=self._combat_move_indices)
                # Run context for value head (bridge doesn't set these)
                sim_state.act_id = self._current_act_id
                sim_state.boss_id = self._current_boss_id
                sim_state.map_path = self._extract_remaining_path(
                    gs, sim_state.floor)
                # Mid-turn counters: bridge.py sets these from the mod API
                # when available (Phase 1A).  Fall back to manual reconstruction
                # only if the mod didn't provide them (old mod version).
                if sim_state.cards_played_this_turn == 0 and cards_played:
                    sim_state.cards_played_this_turn = len(cards_played)
                    skills_this_turn = 0
                    attacks_this_turn = 0
                    for cname in cards_played:
                        if cname.startswith("Use "):
                            continue
                        upgraded = cname.endswith("+")
                        card_def = self.card_db.get_by_name(cname.rstrip("+"), upgraded=upgraded)
                        if card_def and card_def.card_type == CardType.SKILL:
                            skills_this_turn += 1
                        elif card_def and card_def.card_type == CardType.ATTACK:
                            attacks_this_turn += 1
                    if skills_this_turn > 0:
                        sim_state.player.powers["_skills_played"] = skills_this_turn
                    sim_state.attacks_played_this_turn = attacks_this_turn
                # Clear potions already used this turn
                for pidx in self._potions_used_this_turn:
                    if pidx < len(sim_state.player.potions):
                        sim_state.player.potions[pidx] = {}
                hand = list(sim_state.player.hand)
                t0 = time.perf_counter()
                first_action, policy, root_value, child_values, child_visits = self._rust_mcts_search(
                    sim_state, num_simulations=200, temperature=0,
                )
                solve_ms = (time.perf_counter() - t0) * 1000
                total_states += 200
                total_solve_ms += solve_ms
                total_visits = sum(child_visits) if child_visits else 0
                best_visits = max(child_visits) if child_visits else 0
                best_score = best_visits / max(1, total_visits)
                if turn_root_value is None:
                    turn_root_value = root_value
            except Exception as e:
                self._log_action(f"[red]MCTS error: {e}[/red]")
                import traceback
                traceback.print_exc()
                break

            # If MCTS says end turn, we're done
            if first_action.action_type == "end_turn":
                hand_str = ", ".join(c.name for c in hand)
                self._solver_text = (
                    f"[bold]Turn {turn}[/bold] | "
                    f"HP {player.get('current_hp', '?')}/{player.get('max_hp', '?')} | "
                    f"Energy {player.get('energy', '?')}\n"
                    f"Hand: {hand_str}\n"
                    f"vs: {enemy_str}\n\n"
                    f"MCTS: end turn ({solve_ms:.0f}ms)"
                )
                self._review_combat_decision(
                    "End Turn", sim_state, gs, hand, policy,
                    root_value, solve_ms, child_values, child_visits)
                break

            # Handle potion usage from MCTS
            if first_action.action_type == "use_potion":
                # Already used this turn — potion is gone but MCTS doesn't
                # know yet. Log and continue so MCTS re-searches without it.
                if first_action.potion_idx in self._potions_used_this_turn:
                    self._log_action(
                        f"  [yellow]Potion slot {first_action.potion_idx} already used, re-searching[/yellow]"
                    )
                    continue
                pot_name = "potion"
                potions_raw = (gs.get("run") or {}).get("potions", [])
                for p in potions_raw:
                    if p.get("index") == first_action.potion_idx:
                        pot_name = p.get("name", "potion")
                        break
                label = f"Use {pot_name} (slot {first_action.potion_idx})"
                cards_played.append(label)
                targets_chosen.append(first_action.target_idx)
                self._potions_used_this_turn.add(first_action.potion_idx)
                self.logger._emit({
                    "type": "potion_use",
                    "slot": first_action.potion_idx,
                    "potion": pot_name,
                    "target": first_action.target_idx,
                })

                if not self.dry_run:
                    target_info = f" -> enemy {first_action.target_idx}" if first_action.target_idx is not None else ""
                    self._review_combat_decision(
                        f"Use Potion: {pot_name}{target_info}",
                        sim_state, gs, hand, policy,
                        root_value, solve_ms, child_values, child_visits)
                    mcp_action = action_to_mcp(first_action)
                    try:
                        self._execute_with_retry(
                            mcp_action["action"],
                            option_index=mcp_action.get("option_index"),
                            target_index=mcp_action.get("target_index"),
                        )
                        self._log_action(f"  [magenta]>[/magenta] {label}")
                        self.action_count += 1
                    except Exception as e:
                        self._log_action(f"  [red]X {label}: {e}[/red]")
                        break

                    self._refresh()
                    self._wait_for_ready()
                    try:
                        gs = self.client.get_state()
                        self.game_state = gs
                    except Exception:
                        break

                    actions = gs.get("available_actions", [])
                    if "play_card" not in actions:
                        break
                    combat = gs.get("combat") or {}
                    player = combat.get("player") or {}
                    enemies = combat.get("enemies") or []
                continue

            # Resolve card name and target for logging
            if first_action.card_idx is not None and first_action.card_idx < len(hand):
                card = hand[first_action.card_idx]
                target_str = (
                    f" -> enemy {first_action.target_idx}"
                    if first_action.target_idx is not None else ""
                )
                logged_name = f"{card.name}+" if card.upgraded else card.name
                label = f"{logged_name}{target_str}"
                cards_played.append(logged_name)
                targets_chosen.append(first_action.target_idx)
            else:
                label = f"card_idx={first_action.card_idx}"
                cards_played.append(label)
                targets_chosen.append(None)

            if self.dry_run:
                self._log_action(f"  [dim]\\[dry-run] Would play: {label}[/dim]")
                break

            # Execute the single card play
            mcp_action = action_to_mcp(first_action)
            if self.logger:
                enemies_compact = [
                    {"name": e.name, "hp": e.hp, "block": e.block,
                     "intent": e.intent_type, "dmg": e.intent_damage,
                     "hits": e.intent_hits}
                    for e in sim_state.enemies if e.is_alive
                ]
                powers_compact = {k: v for k, v in sim_state.player.powers.items()
                                  if not k.startswith("_") and v != 0}
                self.logger._emit({
                    "type": "mcts_play",
                    "card_idx": first_action.card_idx,
                    "target_idx": first_action.target_idx,
                    "card_name": label,
                    "hand": [c.name for c in hand],
                    "policy": [round(p, 4) for p in policy[:5]],
                    "value": round(root_value, 4),
                    "hp": sim_state.player.hp,
                    "energy": sim_state.player.energy,
                    "block": sim_state.player.block,
                    "powers": powers_compact if powers_compact else None,
                    "enemies": enemies_compact,
                    "attacks_played": sim_state.attacks_played_this_turn,
                })
            # Review pause — show decision before executing
            self._review_combat_decision(
                f"Play: {label}", sim_state, gs, hand, policy,
                root_value, solve_ms, child_values, child_visits)
            try:
                self._execute_with_retry(
                    mcp_action["action"],
                    card_index=mcp_action.get("card_index"),
                    target_index=mcp_action.get("target_index"),
                )
                self._log_action(f"  [green]>[/green] {label}")
                self.action_count += 1
            except Exception as e:
                self._log_action(f"  [red]X {label}: {e}[/red]")
                break

            self._refresh()

            # Wait for game to process, then get fresh state
            self._wait_for_ready()
            try:
                gs = self.client.get_state()
                self.game_state = gs
            except Exception:
                break

            # If we left combat (enemy died, screen changed), stop —
            # UNLESS the game is showing a mid-combat choice (discard,
            # exhaust, pick from pile). Handle those inline via MCTS.
            actions = gs.get("available_actions", [])
            if "play_card" not in actions:
                if "select_deck_card" in actions and self._combat_logged:
                    self._resolve_combat_choice_via_mcts(gs, cards_played, targets_chosen)
                    # After resolving, refresh and continue the play loop
                    try:
                        gs = self.client.get_state()
                        self.game_state = gs
                    except Exception:
                        break
                    actions = gs.get("available_actions", [])
                    if "play_card" not in actions:
                        break
                else:
                    break

            # Update combat locals for next iteration
            combat = gs.get("combat") or {}
            player = combat.get("player") or {}
            enemies = combat.get("enemies") or []

        # Update enemy move indices for next turn's predictions.
        # On first sight, match observed intent to move table; on subsequent
        # turns, just increment (deterministic cycling).
        from .enemy_predict import _match_move_index
        from .simulator import ENEMY_MOVE_TABLES
        for i, e_raw in enumerate(enemies):
            eid = e_raw.get("enemy_id", "")
            key = (i, eid)
            table = ENEMY_MOVE_TABLES.get(eid)
            if not table:
                continue
            if key in self._combat_move_indices:
                self._combat_move_indices[key] = (
                    (self._combat_move_indices[key] + 1) % len(table)
                )
            else:
                from .bridge import parse_intents
                intents = e_raw.get("intents", [])
                it, idmg, ihits, _ = parse_intents(intents)
                idx = _match_move_index(eid, it, idmg, ihits)
                if idx is not None:
                    self._combat_move_indices[key] = idx

        # End turn if we're still in combat
        turn_ended = False
        combat_won_by_card = False
        if not self.dry_run and "end_turn" in gs.get("available_actions", []):
            self._wait_for_ready()
            try:
                self._execute_with_retry("end_turn")
                self._log_action("  [green]>[/green] End Turn")
                self.action_count += 1
                turn_ended = True
            except Exception as e:
                self._log_action(f"  [red]X End Turn: {e}[/red]")
        elif not self.dry_run and "play_card" not in gs.get("available_actions", []):
            # Card play killed the last enemy — combat ended mid-turn.
            # Check if all enemies are dead to confirm it's a win (not a
            # mid-combat discard prompt or phase transition).
            post_enemies = (gs.get("combat") or {}).get("enemies") or []
            all_dead = not post_enemies or all(
                e.get("current_hp", 0) <= 0 for e in post_enemies
            )
            if all_dead:
                combat_won_by_card = True
                turn_ended = True  # Allow turn logging below

        # Only log complete turns (not mid-turn breaks from discard prompts).
        # Mid-turn snapshots are useless for validation and cause false mismatches.
        if not turn_ended and not self.dry_run:
            return

        # Capture hand after all plays (before end turn resolves)
        post_play_hand = None
        try:
            post_combat = (gs.get("combat") or {})
            post_hand_raw = post_combat.get("hand") or []
            post_play_hand = []
            for c in post_hand_raw:
                name = c.get("name") or c.get("card_id", "?")
                if c.get("upgraded"):
                    name += "+"
                post_play_hand.append(name)
        except Exception:
            pass

        # Log the full turn (pass pre-play state for combat snapshot)
        self.logger.log_combat_turn(
            cards_played=cards_played,
            targets_chosen=targets_chosen,
            score=best_score,
            states_evaluated=total_states,
            solve_ms=total_solve_ms,
            game_state=turn_start_gs,
            network_value=turn_root_value,
            hand_after=post_play_hand,
        )

        # Reset accumulator for next turn
        self._turn_cards_played = None
        self._turn_targets_chosen = None
        self._turn_start_gs = None

        if self._store_run_started:
            run_data = (turn_start_gs.get("run") or {})
            combat_data = (turn_start_gs.get("combat") or {})
            player_data = combat_data.get("player") or {}
            self.store.log_combat_turn(
                self._store_run_id,
                floor=run_data.get("floor", 0),
                turn=turn,
                hp=player_data.get("current_hp", 0),
                max_hp=player_data.get("max_hp", 0),
                cards_played=cards_played,
                network_value=turn_root_value,
            )

        self.turn_count += 1

        # Check for combat end — either the last card killed all enemies
        # (combat_won_by_card) or end_turn resolved and the screen changed.
        if combat_won_by_card:
            # Already confirmed all enemies dead above; gs is current state
            self._log_action("[bold green]Combat won![/bold green]")
            self._record_encounter("win")
            self.logger.log_combat_end(gs, "win")
            self._combat_logged = False
            if self._store_run_started:
                post_run = (gs.get("run") or {})
                self.store.log_combat_end(
                    self._store_run_id,
                    floor=post_run.get("floor", 0),
                    hp=post_run.get("current_hp", 0),
                    max_hp=post_run.get("max_hp", 0),
                    outcome="win", turns=self.turn_count,
                )
            self._intercept_card_reward()
        else:
            self._wait_for_ready()
            try:
                post = self.client.get_state()
                post_screen = post.get("screen", "").upper()
                if post_screen not in ("COMBAT",):
                    # Verify enemies are actually dead (not a mid-combat phase transition)
                    combat = post.get("combat") or {}
                    enemies = combat.get("enemies") or []
                    all_dead = not enemies or all(
                        e.get("current_hp", 0) <= 0 for e in enemies
                    )
                    if all_dead:
                        self._log_action("[bold green]Combat won![/bold green]")
                        self._record_encounter("win")
                        self.logger.log_combat_end(post, "win")
                        self._combat_logged = False
                        if self._store_run_started:
                            post_run = post.get("run") or {}
                            self.store.log_combat_end(
                                self._store_run_id,
                                floor=post_run.get("floor", 0),
                                hp=post_run.get("current_hp", 0),
                                max_hp=post_run.get("max_hp", 0),
                                outcome="win", turns=self.turn_count,
                            )
                        self._intercept_card_reward()
            except Exception as e:
                self._log_action(f"[red]Combat end check error: {e}[/red]")
                raise

    # ------------------------------------------------------------------
    # Non-combat
    # ------------------------------------------------------------------

    def _handle_non_combat(self, actions: list[str]) -> None:
        screen_type = detect_screen_type(actions)
        gs = self.game_state
        run = gs.get("run") or {}

        self._log_action(
            f"[blue]Floor {run.get('floor', '?')}[/blue] | {screen_type.upper()}"
        )

        # discard_potion as sole action: game is forcing a potion discard
        # (e.g. potions full after a reward). Just discard slot 0.
        # But this is often a transient state where real actions haven't
        # loaded yet. Skip if: no potions held, OR mid-combat (between
        # turns the game briefly shows only discard_potion before combat
        # actions populate).
        if actions == ["discard_potion"]:
            potions = (run.get("potions") or [])
            has_potions = any(p.get("occupied") for p in potions if isinstance(p, dict))
            if not has_potions:
                self._log_action("  [dim]skip: discard_potion but no potions (transient)[/dim]")
                return
            if self._combat_logged:
                # Mid-combat with only discard_potion — likely transient/stale state.
                # Wait and re-poll rather than acting or crashing.
                retries = getattr(self, "_discard_potion_retries", 0)
                if retries < 10:
                    self._discard_potion_retries = retries + 1
                    if retries == 0:
                        self._log_action("  [dim]waiting: discard_potion mid-combat (stale state)[/dim]")
                    time.sleep(0.5)
                    return
                # After 10 retries (~5s), fall through and execute to avoid permanent stuck
                self._discard_potion_retries = 0
            # Identify which potion is in slot 0
            potion_name = None
            for p in potions:
                if isinstance(p, dict) and p.get("index", p.get("slot")) == 0:
                    potion_name = p.get("name", "unknown")
                    break
            self._log_action(f"  [dim]auto: discard_potion (slot 0: {potion_name})[/dim]")
            self.logger._emit({
                "type": "potion_discard",
                "slot": 0,
                "potion": potion_name,
                "reason": "forced",
                "screen_type": screen_type,
            })
            if not self.dry_run:
                try:
                    self._execute_with_retry("discard_potion", option_index=0)
                    self.action_count += 1
                except Exception as e:
                    self._log_action(f"  [red]Failed to discard potion: {e}[/red]")
            return

        # Reward screen handling.
        # _intercept_card_reward() handles the full reward flow synchronously
        # after combat (non-card rewards first, then card decision).
        # The main tick only sees rewards if the intercept missed or the
        # game re-shows the screen.
        if "choose_reward_card" in actions or "skip_reward_cards" in actions:
            screen_type = "card_reward"
            # Fall through to network handler below

        elif "claim_reward" in actions:
            # Intercept missed some rewards — claim non-card items one at a
            # time (card items are never auto-claimed).
            reward = gs.get("reward") or {}
            if not reward:
                reward = (gs.get("agent_view") or {}).get("reward") or {}
            reward_items = reward.get("rewards") or []
            if not reward_items:
                reward_items = ((gs.get("agent_view") or {}).get("reward") or {}).get("rewards") or []

            full = self._potions_full(gs)
            for item in reward_items:
                if not item.get("claimable", True):
                    continue
                if self._is_card_reward_item(item):
                    continue
                if full and self._is_potion_reward_item(item):
                    continue
                idx = item.get("index", item.get("i"))
                if idx is not None:
                    rtype = str(item.get("reward_type", item.get("line", ""))).split(":")[0]
                    self._log_action(f"  [dim]auto: claim_reward({idx}) — {rtype}[/dim]")
                    if not self.dry_run:
                        try:
                            self._execute_with_retry("claim_reward", option_index=idx)
                            self.action_count += 1
                        except Exception:
                            pass
                    return

            # Only card rewards or nothing left.
            # If the card reward hasn't been handled yet, claim the card
            # item to open card selection instead of skipping.
            if not self._card_reward_handled:
                # Claim the first available reward (index 0) to open
                # card selection.  Don't use stored indices — they shift.
                self._log_action("  [dim]auto: claim_reward(0) — opening card selection[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("claim_reward", option_index=0)
                        self.action_count += 1
                    except Exception:
                        pass
                return  # Next tick will see card selection screen

            if "proceed" in actions:
                self._log_action("  [dim]auto: proceed (rewards done)[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("proceed")
                        self.action_count += 1
                    except Exception:
                        pass
            elif "collect_rewards_and_proceed" in actions:
                if self._card_reward_handled:
                    self._log_action("  [dim]auto: collect_rewards_and_proceed (card handled)[/dim]")
                else:
                    self._log_action("  [dim]auto: collect_rewards_and_proceed[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("collect_rewards_and_proceed")
                        self.action_count += 1
                    except Exception:
                        pass
            return

        elif "collect_rewards_and_proceed" in actions and screen_type != "card_reward":
            # Reward screen with no claim_reward — just proceed.
            if self._card_reward_handled:
                self._log_action("  [dim]auto: collect_rewards_and_proceed (card handled by intercept)[/dim]")
            else:
                self._log_action("  [dim]auto: collect_rewards_and_proceed[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("collect_rewards_and_proceed")
                    self.action_count += 1
                except Exception:
                    pass
            self._track_decision("reward", "auto")
            self.logger.log_decision(
                game_state=gs, screen_type="auto", options=actions,
                choice={"action": "collect_rewards_and_proceed", "option_index": None},
                source="auto",
            )
            return

        # Auto-actions — but prioritize shop opening over proceed
        if screen_type == "auto":
            # If open_shop_inventory is available AND we haven't visited yet,
            # open the shop first (otherwise proceed would skip it entirely).
            # After visiting, _shop_visited is set so we proceed instead.
            if "open_shop_inventory" in actions and not self._shop_visited:
                action_order = ["open_shop_inventory"] + [a for a in actions if a != "open_shop_inventory"]
            else:
                action_order = actions
            for action in action_order:
                if action in AUTO_ACTIONS:
                    self._log_action(f"  [dim]auto: {action}[/dim]")
                    if not self.dry_run:
                        try:
                            self._execute_with_retry(action)
                            self.action_count += 1
                        except Exception as e:
                            self._log_action(f"  [red]Auto-action failed: {e}[/red]")
                    self._track_decision("auto", "auto")
                    self.logger.log_decision(
                        game_state=gs, screen_type="auto", options=actions,
                        choice={"action": action, "option_index": None},
                        source="auto",
                    )
                    return
            return

        # Filter out side-actions that confuse the LLM
        # (discard_potion is always available but is never the primary choice)
        filtered_actions = [a for a in actions if a != "discard_potion"]
        if filtered_actions:
            gs = dict(gs)
            gs["available_actions"] = filtered_actions

        # Deck card select overlay on top of card reward screen:
        # The game can show a deck_card_select preview (e.g. card effect text)
        # while choose_reward_card is also available. If we don't dismiss the
        # overlay first, the card_reward handler gets stuck in a loop.
        if screen_type == "card_reward" and "select_deck_card" in actions:
            sel = gs.get("selection") or {}
            if sel.get("kind") == "deck_card_select":
                # Only auto-dismiss if choose_reward_card is also available,
                # meaning this is truly an overlay on top of the card picker.
                # If select_deck_card is the only action, it's a real card
                # selection — route to the network like any deck_select.
                if "choose_reward_card" in actions:
                    self._log_action("  [dim]auto: select_deck_card(0) — dismiss card preview overlay[/dim]")
                    if not self.dry_run:
                        try:
                            self._execute_with_retry("select_deck_card", option_index=0)
                        except Exception:
                            pass
                    return
                else:
                    self._handle_deck_select(gs)
                    return

        # Multi-select deck screens (e.g. "Choose 2 cards to Add/Remove"):
        # select_deck_card: check if this is a real decision or an
        # informational overlay (e.g. Havoc showing "Draw 3 cards")
        if screen_type == "deck_select":
            if self._deck_select_stuck:
                self._log_action("  [dim]Skipping stuck deck_select screen[/dim]")
                return

            sel = gs.get("selection") or {}
            kind = (sel.get("kind") or "").lower()
            can_confirm = sel.get("confirm", False)

            # Confirm screens — card already selected, just confirm
            if can_confirm:
                self._log_action(f"  [dim]auto: select_deck_card(0) — confirm[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=0)
                    except Exception:
                        pass
                return

            # Route by kind (structured field from game mod).
            # For combat_hand selections, use the prompt to determine
            # whether to pick the worst card (discard/exhaust) or best
            # card (copy/duplicate).
            prompt = strip_markup(sel.get("prompt") or "").lower()

            if kind.startswith("combat_hand"):
                pick_worst = any(kw in prompt for kw in (
                    "discard", "exhaust", "put on top", "draw pile",
                ))
                pick_idx, reasoning = self._pick_combat_hand_card(
                    sel, pick_worst=pick_worst)
                if pick_idx is None:
                    raise RuntimeError(
                        f"No combat_hand pick candidate — "
                        f"kind={kind!r} prompt={prompt!r} "
                        f"cards: {[c.get('name') for c in sel.get('cards', [])]}"
                    )
                self._log_action(f"  [green]{reasoning}[/green]")
                self._track_decision("deck_select", "deterministic")
                if self.logger:
                    self.logger.log_decision(
                        game_state=gs, screen_type="deck_select",
                        options=["select_deck_card"],
                        choice={"action": "select_deck_card",
                                "option_index": pick_idx,
                                "reasoning": reasoning},
                        source="deterministic",
                    )
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=pick_idx)
                    except Exception:
                        pass
                return

            elif kind in ("deck_upgrade_select", "deck_transform_select",
                          "deck_enchant_select", "choose_card_select",
                          "deck_card_select"):
                self._handle_deck_select(gs)
                return

            else:
                raise RuntimeError(
                    f"Unknown deck_select kind: {kind!r} — "
                    f"prompt: {sel.get('prompt', '?')!r}"
                )

        # For finished events or events with only one unlocked option, auto-handle
        if screen_type == "event" and "choose_event_option" in actions:
            event = gs.get("event") or {}
            options = event.get("options") or []
            unlocked = [(i, o) for i, o in enumerate(options) if not o.get("locked")]
            if event.get("finished") or (
                len(options) == 1 and options[0].get("proceed")
            ) or len(unlocked) == 1:
                pick_idx = unlocked[0][0] if unlocked else 0
                label = unlocked[0][1].get("label", "proceed") if unlocked else "proceed"
                self._log_action(f"  [dim]auto: choose_event_option({pick_idx}) — {label}[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("choose_event_option", option_index=pick_idx)
                        self.action_count += 1
                    except Exception as e:
                        self._log_action(f"  [red]Failed: {e}[/red]")
                return

        # For card_reward: skip if already handled (avoid re-presenting to advisor)
        if screen_type == "card_reward" and self._card_reward_handled:
            if "skip_reward_cards" in actions:
                self._log_action("  [dim]auto: skip_reward_cards (already handled)[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("skip_reward_cards")
                    except Exception:
                        pass
            return

        # For card_reward: if card options are empty, skip this tick (data not ready)
        if screen_type == "card_reward":
            reward = gs.get("reward") or {}
            if not reward:
                reward = (gs.get("agent_view") or {}).get("reward") or {}
            card_options = reward.get("card_choices") or reward.get("cards") or []
            sel = gs.get("selection") or {}
            sel_cards = sel.get("cards") or []
            if not card_options and not sel_cards:
                self._log_action("  [dim]Card reward data not ready — waiting[/dim]")
                return

        # General single-option auto-pick: if the screen has exactly one
        # indexed option, pick it without calling the LLM.  Applies to map,
        # event, rest, boss_relic — any screen where there's no real choice.
        _SINGLE_OPT_ACTIONS = {
            "map": "choose_map_node",
            "event": "choose_event_option",
            "rest": "choose_rest_option",
            "boss_relic": "choose_treasure_relic",
        }
        single_action = _SINGLE_OPT_ACTIONS.get(screen_type)
        if single_action and single_action in actions:
            # Count available options from the game state
            option_sources = {
                "map": lambda: (gs.get("map") or {}).get("available_nodes")
                    or (gs.get("map") or {}).get("nodes")
                    or ((gs.get("agent_view") or {}).get("map") or {}).get("available_nodes")
                    or ((gs.get("agent_view") or {}).get("map") or {}).get("nodes")
                    or [],
                "event": lambda: [
                    o for o in ((gs.get("event") or {}).get("options") or [])
                    if not o.get("locked")
                ],
                "rest": lambda: (gs.get("rest") or {}).get("options")
                    or ((gs.get("agent_view") or {}).get("rest") or {}).get("options")
                    or [],
                "boss_relic": lambda: (gs.get("chest") or {}).get("relics")
                    or (gs.get("reward") or {}).get("relics")
                    or ((gs.get("agent_view") or {}).get("chest") or {}).get("relics")
                    or [],
            }
            opts = option_sources.get(screen_type, lambda: [])()
            if len(opts) == 1:
                idx = opts[0].get("index", opts[0].get("i", 0)) if isinstance(opts[0], dict) else 0
                self._log_action(f"  [dim]auto: {single_action}({idx}) — single option[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry(single_action, option_index=idx)
                        self.action_count += 1
                    except Exception as e:
                        self._log_action(f"  [red]Failed: {e}[/red]")
                return

        # Card reward: DeckNet when available, otherwise deterministic advisor.
        # Everything else: deterministic advisor.
        if screen_type == "card_reward" and self._decknet is not None:
            decision = self._decknet_decide_card_reward(gs)
            if decision is not None:
                self._execute_deterministic(
                    gs, decision, screen_type, actions, run,
                )
                return

        _DETERMINISTIC_HANDLERS = {
            "rest": lambda: decide_rest(gs),
            "card_reward": lambda: decide_card_reward(gs, self.game_data),
            "map": lambda: decide_map(gs),
            "shop": lambda: decide_shop(gs, self.game_data),
            "boss_relic": lambda: decide_boss_relic(gs, self.game_data),
            "event": lambda: self._pick_first_unlocked_event_option(gs),
            "bundle": lambda: Decision("choose_bundle", 0, "deterministic: first bundle"),
        }

        handler = _DETERMINISTIC_HANDLERS.get(screen_type)
        if handler:
            decision = handler()
            if decision is None:
                raise RuntimeError(
                    f"Deterministic handler returned None for {screen_type!r}"
                )
            self._execute_deterministic(
                gs, decision, screen_type, actions, run,
            )
            return

        raise RuntimeError(
            f"Unhandled screen type {screen_type!r} — no handler registered. "
            f"Available actions: {actions}"
        )

    # ------------------------------------------------------------------
    # DeckNet card-reward decisions
    # ------------------------------------------------------------------

    def _build_decknet_state(self, gs: dict):
        """Build a DeckBuildingState from live game state for DeckNet inference."""
        from .decknet.state import (
            CardRef, DeckBuildingState, MapRoom, PlayerStats, RoomType,
        )
        from .deterministic_advisor import _get_deck, _gold, _floor

        run = gs.get("run") or {}
        hp = int(run.get("current_hp", 70))
        max_hp = int(run.get("max_hp", 70))
        gold = int(_gold(gs))
        floor = int(_floor(gs))
        potions = len([p for p in run.get("potions") or [] if p])

        deck = []
        for raw in _get_deck(gs):
            cid = (raw.get("card_id") or raw.get("id") or "").rstrip("+")
            if not cid:
                continue
            deck.append(CardRef(id=cid, upgraded=bool(raw.get("upgraded"))))

        relics = frozenset(
            r.get("id", r.get("relic_id", ""))
            for r in (run.get("relics") or [])
            if isinstance(r, dict)
        ) or frozenset(
            r for r in (run.get("relics") or [])
            if isinstance(r, str)
        )

        # Map ahead: convert BFS path of game node types to RoomType enum
        remaining_path = self._extract_remaining_path(gs, floor)
        node_to_room = {
            "Monster": RoomType.NORMAL, "Weak": RoomType.WEAK,
            "Elite": RoomType.ELITE, "Boss": RoomType.BOSS,
            "RestSite": RoomType.REST, "Rest": RoomType.REST,
            "Shop": RoomType.SHOP, "Event": RoomType.EVENT,
            "Treasure": RoomType.TREASURE, "Unknown": RoomType.UNKNOWN,
        }
        map_ahead = [
            MapRoom(
                room_type=node_to_room.get(t, RoomType.UNKNOWN),
                floors_ahead=i,
            )
            for i, t in enumerate(remaining_path[:10])
        ]

        # Map our sim act_id (e.g. "OVERGROWTH") to the numeric act
        act_num = {"OVERGROWTH": 1, "UNDERDOCKS": 2, "HIVE": 2, "GLORY": 3}.get(
            self._current_act_id, 1,
        )

        return DeckBuildingState(
            deck=deck,
            player=PlayerStats(hp=hp, max_hp=max_hp, gold=gold, potions=potions),
            relics=relics,
            act=act_num,
            floor=floor,
            map_ahead=map_ahead,
            boss_id=self._current_boss_id,
        )

    def _decknet_decide_card_reward(self, gs: dict) -> "Decision | None":
        """Use DeckNet to pick the best card reward (including skip).

        Enumerates IDENTITY (skip) + one ADD per offered card, applies each
        mod to the current state, batches them through V, and picks argmax.
        """
        from .decknet.encoder import encode_batch
        from .decknet.state import CardRef, DeckModification, ModKind, apply_mod
        import torch

        reward_data = gs.get("reward") or {}
        av_reward = (gs.get("agent_view") or {}).get("reward") or {}
        rewards = (
            reward_data.get("card_options")
            or av_reward.get("cards")
            or gs.get("card_rewards")
            or gs.get("rewards")
            or (gs.get("selection") or {}).get("cards")
            or []
        )
        if not rewards:
            return None

        state = self._build_decknet_state(gs)

        # Candidates: skip (IDENTITY), then one ADD per offered card
        mods: list[DeckModification] = [DeckModification(kind=ModKind.IDENTITY)]
        card_labels: list[str] = ["skip"]
        card_indices: list[int | None] = [None]
        for i, info in enumerate(rewards):
            cid = (info.get("card_id") or info.get("id") or "").rstrip("+")
            upgraded = bool(info.get("upgraded"))
            name = info.get("name") or cid or "?"
            if not cid:
                continue
            mods.append(DeckModification(
                kind=ModKind.ADD, card=CardRef(id=cid, upgraded=upgraded),
            ))
            card_labels.append(name)
            card_indices.append(info.get("index", i))

        candidate_states = [apply_mod(state, m) for m in mods]
        batch = encode_batch(candidate_states, self._decknet_card_vocab)
        with torch.no_grad():
            values = self._decknet(
                batch["card_ids"], batch["card_stats"],
                batch["deck_mask"], batch["global_state"],
            )
        best = int(torch.argmax(values).item())
        value_list = values.tolist()

        hs = {
            "head": "decknet_v",
            "chosen": best,
            "options": [
                {"label": lbl, "score": round(v, 4)}
                for lbl, v in zip(card_labels, value_list)
            ],
        }

        if best == 0:
            return Decision(
                "skip_reward_cards", None,
                f"DeckNet: skip (V={value_list[0]:+.3f})",
                network_value=float(value_list[0]), head_scores=hs,
            )
        return Decision(
            "choose_reward_card", card_indices[best],
            f"DeckNet: take {card_labels[best]} (V={value_list[best]:+.3f})",
            network_value=float(value_list[best]), head_scores=hs,
        )

    def _pick_first_unlocked_event_option(self, gs: dict) -> "Decision":
        """Deterministic event handler: pick the first unlocked option."""
        event = gs.get("event") or (gs.get("agent_view") or {}).get("event") or {}
        options = event.get("options") or []
        for i, opt in enumerate(options):
            if not opt.get("locked"):
                idx = opt.get("index", i)
                label = opt.get("title") or opt.get("name") or f"option {idx}"
                return Decision(
                    "choose_event_option", idx,
                    f"deterministic: first unlocked ({label})",
                )
        return Decision("choose_event_option", 0, "deterministic: fallback 0")

    def _pick_combat_hand_card(
        self, sel: dict, pick_worst: bool,
    ) -> "tuple[int | None, str]":
        """Pick a card from hand for a mid-combat selection screen.

        For discard/exhaust/put-on-top (pick_worst=True): prefer Strike,
        then Defend, else first card. For keep/duplicate (pick_worst=False):
        prefer non-Strike/Defend cards, else first card.
        """
        cards = sel.get("cards", [])
        if not cards:
            return None, "no cards in selection"

        def base_name(info: dict) -> str:
            return (info.get("name") or info.get("card_id") or "").rstrip("+")

        def is_basic(info: dict) -> bool:
            name = base_name(info)
            return "Strike" in name or "Defend" in name

        if pick_worst:
            for info in cards:
                if is_basic(info):
                    idx = info.get("index", cards.index(info))
                    return idx, f"deterministic: discard {base_name(info)}"
            first = cards[0]
            return first.get("index", 0), f"deterministic: discard {base_name(first)}"

        for info in cards:
            if not is_basic(info):
                idx = info.get("index", cards.index(info))
                return idx, f"deterministic: keep {base_name(info)}"
        first = cards[0]
        return first.get("index", 0), f"deterministic: keep {base_name(first)}"

    def _detect_run_context(self, gs: dict) -> None:
        """Detect act_id and boss_id from game state at run start."""
        from .game_data import strip_markup

        # Try to get act from game state
        run = gs.get("run") or {}
        act_name = run.get("act") or run.get("act_name") or ""
        act_map = {
            "Overgrowth": "OVERGROWTH", "Act 1 - Overgrowth": "OVERGROWTH",
            "Underdocks": "UNDERDOCKS",
            "Hive": "HIVE", "Act 2 - Hive": "HIVE",
            "Glory": "GLORY", "Act 3 - Glory": "GLORY",
        }
        self._current_act_id = act_map.get(act_name, "")

        # Try to get boss from map's boss node
        # Boss vocab uses encounter IDs like "CEREMONIAL_BEAST_BOSS"
        map_data = gs.get("map") or (gs.get("agent_view") or {}).get("map") or {}
        boss_node = map_data.get("boss_node") or {}
        boss_name = boss_node.get("name") or boss_node.get("encounter") or ""
        if boss_name:
            boss_id = boss_name.upper().replace(" ", "_")
            if not boss_id.endswith("_BOSS"):
                boss_id += "_BOSS"
            self._current_boss_id = boss_id
        else:
            self._current_boss_id = ""

    def _extract_remaining_path(self, gs: dict, current_floor: int) -> tuple[str, ...]:
        """BFS from the player's current map node to get downstream room types."""
        from .simulator import _bfs_downstream_path

        map_data = gs.get("map") or (gs.get("agent_view") or {}).get("map") or {}
        nodes = map_data.get("nodes") or []
        if not nodes:
            return ()

        # Find the player's current node
        current = map_data.get("current_node") or {}
        current_pos = (current.get("row"), current.get("col"))

        by_pos = {(n["row"], n["col"]): n for n in nodes
                  if "row" in n and "col" in n}
        current_node = by_pos.get(current_pos)
        if not current_node:
            return ()

        return _bfs_downstream_path(
            {"nodes": nodes}, current_node, max_depth=10)

    def _resolve_combat_choice_via_mcts(
        self, gs: dict,
        cards_played: list[str], targets_chosen: list[int | None],
    ) -> None:
        """Handle mid-combat choices (Survivor discard, Acrobatics, etc.) via MCTS.

        When a card creates a pending choice, MCTS naturally explores each
        option as a branch in the search tree. This runs a fresh MCTS search
        on the current state with a pending_choice set, letting MCTS evaluate
        the tactical consequences of each choice (what to draw, block vs damage
        tradeoffs, etc.) rather than using the strategic OPTION_SHOP_REMOVE
        workaround.
        """
        from .models import PendingChoice

        sel = gs.get("selection") or {}
        kind = (sel.get("kind") or "").lower()
        prompt = strip_markup(sel.get("prompt") or "").lower()
        sel_cards = sel.get("cards") or []

        if not sel_cards:
            self._log_action("  [yellow]Combat choice: no cards in selection, falling back[/yellow]")
            return

        # Determine choice type from selection kind and prompt
        if "discard" in prompt or "exhaust" in prompt:
            choice_type = "discard_from_hand"
        elif "draw pile" in prompt or "put on top" in prompt:
            choice_type = "discard_from_hand"  # mechanically same: pick card from hand
        elif "choose" in prompt and "hand" in prompt:
            choice_type = "choose_from_hand"
        else:
            choice_type = "discard_from_hand"  # default for combat_hand kinds

        # Determine source card (the card that triggered this choice)
        source_card_id = ""
        if cards_played:
            last_played = cards_played[-1]
            if not last_played.startswith("Use "):
                card_def = self.card_db.get_by_name(last_played.rstrip("+"))
                if card_def:
                    source_card_id = card_def.id

        # Determine num_choices from selection data
        num_choices = sel.get("num_choices", sel.get("max_choices", 1))

        # Build sim state from fresh game state
        try:
            sim_state = state_from_mcp(gs, self.card_db,
                                       move_indices=self._combat_move_indices)
        except Exception as e:
            self._log_action(f"  [red]Failed to build sim state for choice: {e}[/red]")
            return

        # Build a mapping from hand card identity to game selection indices
        # so we can translate MCTS's choice back to the game's option_index.
        game_idx_by_card: dict[tuple[str, int], int] = {}
        for ci, card_info in enumerate(sel_cards):
            card_id = card_info.get("card_id") or card_info.get("id", "")
            game_idx = card_info.get("index", ci)
            # Use (card_id, occurrence_count) as key to handle duplicates
            occ = sum(1 for k in game_idx_by_card if k[0] == card_id)
            game_idx_by_card[(card_id, occ)] = game_idx

        # Set pending choice on sim state
        sim_state.pending_choice = PendingChoice(
            choice_type=choice_type,
            num_choices=num_choices,
            source_card_id=source_card_id,
        )

        # Run MCTS — enumerate_actions will return only choose_card actions
        try:
            first_action, policy, root_value, _, _ = self._rust_mcts_search(
                sim_state, num_simulations=200, temperature=0,
            )
        except Exception as e:
            self._log_action(f"  [red]MCTS choice search failed: {e}[/red]")
            return

        if first_action.action_type != "choose_card" or first_action.choice_idx is None:
            self._log_action(
                f"  [yellow]MCTS returned {first_action.action_type} for choice, "
                f"expected choose_card[/yellow]"
            )
            return

        # Map MCTS choice_idx (hand index) to game option_index
        chosen_card = sim_state.player.hand[first_action.choice_idx]
        chosen_id = chosen_card.id.rstrip("+")
        # Count occurrences of this card_id before the chosen index to handle dupes
        occ = sum(1 for i in range(first_action.choice_idx)
                  if sim_state.player.hand[i].id.rstrip("+") == chosen_id)
        game_idx = game_idx_by_card.get((chosen_id, occ))

        if game_idx is None:
            # Fallback: try matching by name
            for ci, card_info in enumerate(sel_cards):
                if card_info.get("name") == chosen_card.name:
                    game_idx = card_info.get("index", ci)
                    break

        if game_idx is None:
            self._log_action(
                f"  [red]Could not map MCTS choice {chosen_card.name} "
                f"(idx={first_action.choice_idx}) to game index[/red]"
            )
            return

        verb = "discard" if "discard" in choice_type else "choose"
        self._log_action(
            f"  [blue]MCTS {verb}: {chosen_card.name} "
            f"(value={root_value:.2f})[/blue]"
        )

        # Review pause — show all options with MCTS scores
        if self.review_mode:
            option_labels = [c.get("name", "?") for c in sel_cards]
            lines = []
            for i, (lbl, p) in enumerate(zip(option_labels, policy)):
                marker = " *" if i == first_action.choice_idx else ""
                lines.append(f"  {lbl}  visits={p:.0%}{marker}")
            safe_prompt = prompt.title().replace("[", "(").replace("]", ")")
            self._review_pause(
                f"[bold]{safe_prompt}:[/bold] MCTS {verb} {chosen_card.name} "
                f"(value={root_value:+.2f})\n"
                + "\n".join(lines)
            )

        # Log the decision
        if self.logger:
            option_labels = [c.get("name", "?") for c in sel_cards]
            self.logger.log_decision(
                game_state=gs, screen_type="deck_select",
                options=["select_deck_card"],
                choice={"action": "select_deck_card",
                        "option_index": game_idx,
                        "reasoning": f"MCTS {verb} {chosen_card.name} (value={root_value:.2f})"},
                source="network",
                network_value=root_value,
                head_scores={
                    "head": "mcts_choice",
                    "chosen": first_action.choice_idx,
                    "options": [
                        {"label": lbl, "score": round(p, 4)}
                        for lbl, p in zip(option_labels, policy)
                    ],
                },
            )

        # Execute the choice
        if not self.dry_run:
            try:
                self._execute_with_retry("select_deck_card", option_index=game_idx)
                self.action_count += 1
            except Exception as e:
                self._log_action(f"  [red]Failed to submit choice: {e}[/red]")

        # For multi-select, handle additional choices recursively
        if num_choices > 1:
            self._wait_for_ready()
            try:
                gs2 = self.client.get_state()
                self.game_state = gs2
                actions2 = gs2.get("available_actions", [])
                if "select_deck_card" in actions2 and self._combat_logged:
                    self._resolve_combat_choice_via_mcts(gs2, cards_played, targets_chosen)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Deterministic decision execution
    # ------------------------------------------------------------------

    def _execute_deterministic(
        self,
        gs: dict,
        decision,  # deterministic_advisor.Decision
        screen_type: str,
        actions: list[str],
        run: dict,
    ) -> None:
        """Execute a deterministic advisor decision."""
        from .deterministic_advisor import Decision

        # Validate action is available
        if decision.action not in actions:
            self._log_action(
                f"  [yellow]Deterministic action '{decision.action}' not available, "
                f"falling back[/yellow]"
            )
            # Fallback by screen type
            _FALLBACKS = {
                "rest": ("choose_rest_option", 0),
                "card_reward": ("skip_reward_cards", None),
                "map": ("choose_map_node", 0),
                "shop": ("close_shop_inventory", None),
                "boss_relic": ("choose_treasure_relic", 0),
                "deck_select": ("select_deck_card", 0),
                "event": ("choose_event_option", 0),
                "bundle": ("choose_bundle", 0),
            }
            fb = _FALLBACKS.get(screen_type)
            if fb and fb[0] in actions:
                self._log_action(
                    f"  [bold yellow]WARNING: Using fallback {fb[0]}({fb[1]}) "
                    f"for {screen_type} — network/deterministic action was unavailable[/bold yellow]"
                )
                decision = Decision(fb[0], fb[1], "fallback")
            else:
                return

        # Log the decision
        self._log_action(
            f"  [blue]Decision: {decision.action}"
            f"{f' ({decision.option_index})' if decision.option_index is not None else ''}"
            f" — {decision.reasoning}[/blue]"
        )

        source = "network" if decision.network_value is not None else "deterministic"
        self._track_decision(screen_type, source)

        if self.logger:
            self.logger.log_decision(
                game_state=gs,
                screen_type=screen_type,
                options=actions,
                choice={
                    "action": decision.action,
                    "option_index": decision.option_index,
                    "reasoning": decision.reasoning,
                },
                source=source,
                network_value=decision.network_value,
                head_scores=decision.head_scores,
            )

        if self._store_run_started:
            self.store.log_decision(
                self._store_run_id,
                floor=run.get("floor", 0),
                hp=run.get("current_hp", 0),
                max_hp=run.get("max_hp", 0),
                screen_type=screen_type,
                choice=decision.reasoning,
                network_value=decision.network_value,
                head_scores=decision.head_scores,
            )

        # Execute
        if not self.dry_run:
            # Review pause — show non-combat decision before applying
            head_vals = ""
            scores_str = ""
            if decision.head_scores and "options" in decision.head_scores:
                opts = decision.head_scores["options"]
                scores_str = "\n  " + " | ".join(
                    f"{o['label']}: {o['score']:.2f}" for o in opts
                )
            self._review_pause(
                f"[bold]{screen_type.upper()}:[/bold] {decision.action}"
                f"{f' (idx={decision.option_index})' if decision.option_index is not None else ''}\n"
                f"  {decision.reasoning}\n"
                f"  Floor {run.get('floor', '?')} | "
                f"HP {run.get('current_hp', '?')}/{run.get('max_hp', '?')}\n"
                f"{head_vals}{scores_str}"
            )
            try:
                self._execute_with_retry(
                    decision.action, option_index=decision.option_index,
                )
            except Exception as e:
                self._log_action(f"  [red]Execution failed: {e}[/red]")
                return

        # Post-execution bookkeeping
        if screen_type == "card_reward":
            self._card_reward_handled = True
            if decision.action == "skip_reward_cards":
                # Record deck size so we can detect if a card gets added anyway
                self._deck_size_after_skip = len(
                    (gs.get("run") or {}).get("deck", [])
                )
        if screen_type == "shop" and decision.action == "close_shop_inventory":
            self._shop_visited = True

        # Update advisor panel
        source_label = "network" if decision.network_value is not None else "deterministic"
        source_color = "blue" if decision.network_value is not None else "green"
        self._advisor_text = (
            f"[bold]{screen_type.upper()}[/bold] | "
            f"Floor {run.get('floor', '?')} | "
            f"HP {run.get('current_hp', '?')}/{run.get('max_hp', '?')}\n\n"
            f"[{source_color}]\\[{source_label}][/{source_color}] {decision.action}"
            f"{f' (idx={decision.option_index})' if decision.option_index is not None else ''}\n"
            f"{decision.reasoning}"
        )

        self.action_count += 1
        self._refresh()

    # ------------------------------------------------------------------
    # Multi-select deck screens
    # ------------------------------------------------------------------

    def _handle_deck_select(self, gs: dict) -> None:
        """Handle deck card selection screens (add, remove, upgrade, transform).

        For single-select (upgrade, transform): use the advisor.
        For multi-select (e.g. "Choose 2 to Remove"): pick deterministically
        using Strikes first, then Defends, never key card — since the advisor
        would give the same answer every call and multi-select toggling is
        unreliable via the API.
        """
        sel = gs.get("selection") or {}
        prompt_text = strip_markup(sel.get("prompt") or "").lower()
        cards = sel.get("cards", [])

        # Detect multi-select from prompt (e.g. "Choose 2 cards to Remove")
        import re
        multi_match = re.search(r"choose\s+(\d+)", prompt_text)
        is_multi = multi_match is not None and int(multi_match.group(1)) > 1

        if is_multi:
            self._handle_multi_deck_select(gs, cards, prompt_text)
        else:
            self._handle_single_deck_select(gs)

    def _handle_single_deck_select(self, gs: dict) -> None:
        """Single-select deck screen — uses the deterministic advisor."""
        decision = decide_deck_select(gs)
        actions = gs.get("available_actions", [])
        run = gs.get("run") or {}
        self._execute_deterministic(gs, decision, "deck_select", actions, run)

        # Wait for screen to change or confirm
        time.sleep(0.5)
        try:
            gs = self.client.get_state()
        except Exception:
            return
        self.game_state = gs
        if "confirm_selection" in gs.get("available_actions", []):
            if not self.dry_run:
                try:
                    self._execute_with_retry("confirm_selection")
                except Exception:
                    pass
            self._log_action("  [dim]auto: confirm_selection[/dim]")

    def _rank_deck_cards_deterministic(
        self, gs: dict,
    ) -> "list[tuple[int, str, float]]":
        """Rank deck_select cards for multi-select using deterministic advisor logic.

        Returns list of (game_index, card_name, score) sorted best-first for
        the detected operation (upgrade wants highest-value, remove wants
        lowest-value). The "score" is just a rank proxy for logging.
        """
        from .config import CARD_TIERS, CHARACTER_CONFIG, detect_character
        from .deterministic_advisor import _card_tier

        sel = gs.get("selection") or {}
        prompt = strip_markup(sel.get("prompt") or "").lower()
        cards = sel.get("cards", [])
        if not cards:
            return []

        character = detect_character(gs)
        cfg = CHARACTER_CONFIG.get(character, CHARACTER_CONFIG["ironclad"])
        protect = set(cfg.get("protect_cards", [cfg["key_card"]]))

        is_remove = "remove" in prompt or "destroy" in prompt or "transform" in prompt

        tier_rank = {"S": 100, "A": 70, "B": 30, "avoid": 0}
        scored: list[tuple[int, str, float]] = []
        for i, info in enumerate(cards):
            name = info.get("name", info.get("card_id", "?"))
            idx = info.get("index", i)
            base = name.rstrip("+")
            if is_remove:
                if name in protect:
                    score = 100.0
                elif "Strike" in base or "Defend" in base:
                    score = 0.0
                else:
                    tier = _card_tier(base, character)
                    score = float(tier_rank.get(tier, 50))
                scored.append((idx, name, -score))
            else:
                tier = _card_tier(base, character)
                score = float(tier_rank.get(tier, 15))
                if info.get("type", "").lower() == "power":
                    score += 10.0
                scored.append((idx, name, score))

        scored.sort(key=lambda x: x[2], reverse=True)
        return scored

    def _handle_multi_deck_select(self, gs: dict, cards: list, prompt_text: str) -> None:
        """Multi-select deck screen — pick the top N cards via deterministic ranking."""
        import re
        multi_match = re.search(r"choose\s+(\d+)", prompt_text)
        num_to_pick = int(multi_match.group(1)) if multi_match else 2

        ranked = self._rank_deck_cards_deterministic(gs)
        priority = [idx for idx, _, _ in ranked]

        ranking_str = ", ".join(
            f"{name}={score:.1f}" for _, name, score in ranked
        )
        self._log_action(
            f"  [green]Multi-select ({num_to_pick}/{len(cards)}): "
            f"deterministic ranking: {ranking_str}[/green]"
        )
        if self.logger:
            for pick_num in range(min(num_to_pick, len(ranked))):
                pick_idx, pick_name, pick_score = ranked[pick_num]
                self.logger.log_decision(
                    game_state=gs,
                    screen_type="deck_select",
                    options=["select_deck_card"],
                    choice={
                        "action": "select_deck_card",
                        "option_index": pick_idx,
                        "reasoning": f"deterministic multi-select {pick_num+1}/{num_to_pick}: "
                                     f"{pick_name} (score={pick_score:.1f})",
                    },
                    source="deterministic",
                )
        self._track_decision("deck_select", "deterministic")

        picked = 0
        max_attempts = num_to_pick * 3  # Safety limit

        for attempt in range(max_attempts):
            if picked >= num_to_pick:
                break
            if attempt >= len(priority):
                break

            idx = priority[attempt] if attempt < len(priority) else attempt
            card_name = next(
                (c.get("name", "?") for c in cards if c.get("index") == idx),
                f"index {idx}",
            )
            self._log_action(f"  [cyan]Selecting {card_name} (index {idx})[/cyan]")

            if not self.dry_run:
                try:
                    self._execute_with_retry("select_deck_card", option_index=idx)
                except Exception as e:
                    self._log_action(f"  [yellow]Select failed: {e}[/yellow]")
                    continue

            time.sleep(1.0)  # Wait for the game to process

            try:
                gs = self.client.get_state()
            except Exception:
                return
            self.game_state = gs
            actions = gs.get("available_actions", [])

            # Check if confirm appeared or screen changed
            if "confirm_selection" in actions:
                if not self.dry_run:
                    try:
                        self._execute_with_retry("confirm_selection")
                    except Exception:
                        pass
                self._log_action("  [dim]auto: confirm_selection[/dim]")
                return
            if "select_deck_card" not in actions:
                return  # Screen changed, done

            # Check selected count
            sel = gs.get("selection") or {}
            curr = sel.get("selected_count", 0)
            if curr > picked:
                picked = curr

        # If we exhausted attempts without the screen changing,
        # mark as stuck so the main loop can time out
        if picked < num_to_pick:
            self._log_action(f"  [yellow]Multi-select stuck (picked {picked}/{num_to_pick})[/yellow]")
            self._deck_select_stuck = True
            self._stuck_since = time.monotonic()

    # ------------------------------------------------------------------
    # Game over
    # ------------------------------------------------------------------

    def _record_encounter(self, outcome: str) -> None:
        """Track HP lost per combat. The worst encounter of the run is saved at game over."""
        enc = getattr(self, "_combat_encounter", None)
        if enc is None:
            return

        # Compute HP lost this combat
        start_hp = getattr(self, "_combat_start_hp", enc.get("player_hp", 70))
        if outcome == "win":
            gs = self.game_state
            post_hp = (gs.get("run") or {}).get("current_hp", start_hp)
        else:
            post_hp = 0
        hp_lost = start_hp - post_hp

        record = {**enc, "outcome": outcome, "turns": self.turn_count,
                  "hp_lost": hp_lost, "hp_after": post_hp}

        # Track worst encounter this run (most HP lost)
        if not hasattr(self, "_worst_encounter") or self._worst_encounter is None:
            self._worst_encounter = record
        elif hp_lost > self._worst_encounter.get("hp_lost", 0):
            self._worst_encounter = record

    def _save_worst_encounter(self) -> None:
        """At end of run, save the encounter where we lost the most HP."""
        import json as _json
        rec = getattr(self, "_worst_encounter", None)
        if rec is None:
            return
        self._worst_encounter = None

        self._log_action(
            f"[yellow]Worst encounter: {rec['enemy_names']} "
            f"(lost {rec['hp_lost']} HP, floor {rec['floor']})[/yellow]"
        )

        rec["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        from pathlib import Path as _P
        path = _P(__file__).resolve().parents[3] / "betaone_checkpoints" / "recorded_encounters.jsonl"
        with open(path, "a", encoding="utf-8") as f:
            f.write(_json.dumps(rec) + "\n")

    def _handle_game_over(self) -> None:
        gs = self.game_state
        run = gs.get("run") or {}
        game_over = gs.get("game_over") or {}
        outcome = game_over.get("outcome", "unknown")
        floor = run.get("floor", "?")
        hp = run.get("current_hp", 0)

        if outcome == "victory" or hp > 0:
            self._log_action(
                f"[bold green]VICTORY![/bold green] Floor {floor} | HP {hp}"
            )
            self.logger.log_combat_end(gs, "win")
            self._combat_logged = False
            self.logger.log_run_end(gs, "victory")
            result = "victory"
        else:
            self._log_action(
                f"[bold red]DEFEAT[/bold red] Floor {floor} | HP {hp}"
            )
            self._record_encounter("defeat")
            self.logger.log_combat_end(gs, "defeat")
            self._combat_logged = False
            self.logger.log_run_end(gs, "defeat")
            result = "defeat"

        # Save the worst encounter from this run (most HP lost)
        self._save_worst_encounter()

        if self._store_run_started:
            self.store.log_combat_end(
                self._store_run_id,
                floor=floor if isinstance(floor, int) else 0,
                hp=hp, max_hp=run.get("max_hp", 0),
                outcome="win" if result == "victory" else "defeat",
                turns=self.turn_count,
            )
            self.store.end_run(
                self._store_run_id, outcome=result,
                floor=floor if isinstance(floor, int) else 0,
                hp=hp, max_hp=run.get("max_hp"),
            )
            self._store_run_started = False

        self._log_action(
            f"Turns: {self.turn_count} | Actions: {self.action_count}"
        )

        # Emit decision routing summary so we can spot network bypasses
        self._emit_routing_summary(result, floor)
        self._refresh()

        # Return to main menu so the next run can start
        actions = gs.get("available_actions", [])
        if "return_to_main_menu" in actions and not self.dry_run:
            time.sleep(1.0)
            try:
                self._execute_with_retry("return_to_main_menu")
                time.sleep(2.0)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Reward interception
    # ------------------------------------------------------------------

    def _intercept_card_reward(self) -> None:
        """Handle the reward screen after combat using resolve_rewards.

        Polls until the reward/card-selection screen appears, asks the
        network which card to take (or skip), then calls the mod's
        resolve_rewards action which claims gold/potion/relic and
        takes/skips the card atomically.
        """
        deadline = time.monotonic() + 5.0
        polls = 0

        while time.monotonic() < deadline:
            time.sleep(0.1)
            polls += 1
            try:
                gs = self.client.get_state()
            except Exception:
                return

            screen = gs.get("screen", "").upper()
            actions = gs.get("available_actions", [])

            if polls <= 3:
                self._log_action(
                    f"  [dim]reward poll #{polls}: screen={screen} actions={actions[:4]}[/dim]"
                )

            # Reward or card selection screen — resolve atomically
            if "REWARD" in screen or "CARD_SELECTION" in screen:
                self._resolve_rewards_atomic(gs)
                return

            # Left the reward screen entirely — nothing to do
            if screen not in ("", "COMBAT"):
                self._log_action(
                    f"  [yellow]Left reward screen (screen={screen})[/yellow]"
                )
                return

    def _resolve_rewards_atomic(self, gs: dict) -> None:
        """Use network to decide card reward, then claim rewards.

        On NCardRewardSelectionScreen (card_options visible): evaluate with
        the network and call resolve_rewards atomically.

        On NRewardsScreen (card_options empty): card options aren't visible
        yet.  Claim non-card rewards individually, then open the card
        selection screen via claim_reward on the card item, wait for card
        options, and let the network decide.
        """
        self.game_state = gs

        reward_data = gs.get("reward") or {}
        av_reward = (gs.get("agent_view") or {}).get("reward") or {}
        card_options = (
            reward_data.get("card_options")
            or av_reward.get("cards")
            or (gs.get("selection") or {}).get("cards")
            or []
        )

        if card_options:
            # Already on card selection screen — decide and resolve atomically
            card_index = self._evaluate_card_reward(gs, card_options)
            # Log non-card reward items that will be claimed atomically
            reward_items = reward_data.get("rewards") or av_reward.get("rewards") or []
            if reward_items:
                self.logger._emit({
                    "type": "reward_claim_atomic",
                    "items": reward_items,
                    "card_index": card_index,
                })
            self._log_action(
                f"  [green]resolve_rewards (card_index={card_index})[/green]"
            )
            self._card_reward_handled = True
            if not self.dry_run:
                try:
                    if card_index is not None:
                        self._execute_with_retry(
                            "resolve_rewards", card_index=card_index)
                    else:
                        self._execute_with_retry("resolve_rewards")
                    self.action_count += 1
                except Exception as e:
                    self._log_action(f"  [red]resolve_rewards failed: {e}[/red]")
                    try:
                        self._execute_with_retry("collect_rewards_and_proceed")
                    except Exception:
                        pass
            return

        # On NRewardsScreen — card options aren't visible yet.
        # Claim non-card rewards one at a time, then wait for the card
        # selection screen.  Skip potion rewards when potion slots are full
        # (claiming would trigger a discard_potion screen and derail the flow).
        self._log_action("  [dim]On reward screen — claiming rewards to reach card selection[/dim]")

        if not self.dry_run:
            for _ in range(10):
                try:
                    fresh = self.client.get_state()
                except Exception:
                    break
                self.game_state = fresh
                fresh_screen = fresh.get("screen", "").upper()

                # Reached card selection — stop claiming
                if "CARD_SELECTION" in fresh_screen:
                    break
                # Left rewards entirely — no card reward exists
                if "REWARD" not in fresh_screen:
                    break

                fresh_reward = fresh.get("reward") or {}
                fresh_items = fresh_reward.get("rewards") or []
                if not fresh_items:
                    break

                # Find first claimable item (skip cards, skip potions when full)
                full = self._potions_full(fresh)
                claim_idx = None
                claim_item = None
                for i, item in enumerate(fresh_items):
                    if not isinstance(item, dict):
                        claim_idx = i
                        claim_item = item
                        break
                    if self._is_card_reward_item(item):
                        continue
                    if full and self._is_potion_reward_item(item):
                        continue
                    claim_idx = item.get("index", i)
                    claim_item = item
                    break

                if claim_idx is None:
                    # Only card rewards and/or uncollectable potions remain.
                    # Claim the card reward item to open card selection.
                    for i2, item2 in enumerate(fresh_items):
                        if isinstance(item2, dict) and self._is_card_reward_item(item2):
                            cidx = item2.get("index", i2)
                            self._log_action(f"  [dim]claim_reward[{cidx}]: opening card selection[/dim]")
                            try:
                                self._execute_with_retry("claim_reward", option_index=cidx)
                                self.action_count += 1
                                time.sleep(0.3)
                            except Exception:
                                pass
                            break
                    break

                item_desc = claim_item if isinstance(claim_item, str) else claim_item.get("name", claim_item.get("type", "?"))
                self._log_action(f"  [dim]claim_reward[{claim_idx}]: {item_desc}[/dim]")
                self.logger._emit({
                    "type": "reward_claim",
                    "item": claim_item if isinstance(claim_item, (str, dict)) else str(claim_item),
                })

                try:
                    self._execute_with_retry("claim_reward", option_index=claim_idx)
                    self.action_count += 1
                    time.sleep(0.2)
                except Exception:
                    break

        # Poll for card options on the card selection screen
        card_options = []
        deadline = time.monotonic() + 5.0
        poll_count = 0
        while time.monotonic() < deadline:
            time.sleep(0.3)
            try:
                gs = self.client.get_state()
            except Exception:
                break
            self.game_state = gs
            screen = gs.get("screen", "").upper()

            reward_data = gs.get("reward") or {}
            av_reward = (gs.get("agent_view") or {}).get("reward") or {}
            card_options = (
                reward_data.get("card_options")
                or av_reward.get("cards")
                or (gs.get("selection") or {}).get("cards")
                or []
            )
            poll_count += 1
            self._log_action(
                f"  [dim]reward poll #{poll_count}: screen={screen} "
                f"actions={gs.get('available_actions', [])} "
                f"card_options={len(card_options)}[/dim]"
            )
            if card_options:
                break
            # Left the reward flow entirely (no card reward existed)
            if "REWARD" not in screen and "CARD_SELECTION" not in screen:
                break
            # Still on REWARD with claim_reward available — card reward item
            # wasn't detected by _is_card_reward_item.  Claim index 0 as
            # a fallback to open card selection.
            if (not card_options
                    and "claim_reward" in gs.get("available_actions", [])
                    and poll_count == 1):
                self._log_action("  [dim]claim_reward[0]: fallback open card selection[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("claim_reward", option_index=0)
                        self.action_count += 1
                        time.sleep(0.3)
                    except Exception:
                        pass

        # Step 4: Decide and pick/skip
        if card_options:
            self._card_reward_handled = True
        if card_options and not self.dry_run:
            card_index = self._evaluate_card_reward(gs, card_options)
            if card_index is not None:
                self._log_action(
                    f"  [green]choose_reward_card (card_index={card_index})[/green]"
                )
                try:
                    self._execute_with_retry(
                        "choose_reward_card", option_index=card_index)
                    self.action_count += 1
                except Exception as e:
                    self._log_action(f"  [red]choose_reward_card failed: {e}[/red]")
            else:
                self._log_action("  [dim]skip_reward_cards[/dim]")
                try:
                    self._execute_with_retry("skip_reward_cards")
                    self.action_count += 1
                except Exception as e:
                    self._log_action(f"  [red]skip_reward_cards failed: {e}[/red]")
        elif not card_options:
            self._log_action("  [dim]No card options found — proceeding[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("collect_rewards_and_proceed")
                except Exception:
                    pass

    def _evaluate_card_reward(
        self, gs: dict, card_options: list,
    ) -> int | None:
        """Pick a card reward via DeckNet, or deterministic advisor as fallback.

        Returns card_index to take, or None to skip.
        """
        if self._decknet is not None:
            decision = self._decknet_decide_card_reward(gs)
            source = "decknet"
        else:
            decision = decide_card_reward(gs, self.game_data)
            source = "deterministic"

        if decision is None:
            return None

        self._log_action(f"  [blue]{decision.reasoning}[/blue]")
        self._track_decision("card_reward", source)
        if self.review_mode and decision.head_scores:
            opts = decision.head_scores.get("options", [])
            scores_str = "\n".join(
                f"  {o['label']}  score={o['score']:+.3f}" for o in opts)
            self._review_pause(
                f"[bold]CARD REWARD:[/bold] {decision.reasoning}\n"
                f"{scores_str}"
            )
        if self.logger:
            self.logger.log_decision(
                game_state=gs, screen_type="card_reward",
                options=gs.get("available_actions", []),
                choice={"action": decision.action,
                        "option_index": decision.option_index,
                        "reasoning": decision.reasoning},
                source=source,
                network_value=decision.network_value,
                head_scores=decision.head_scores,
            )
        if decision.action == "choose_reward_card":
            return decision.option_index
        self._deck_size_after_skip = len(
            (gs.get("run") or {}).get("deck", [])
        )
        return None

    # ------------------------------------------------------------------
    # Action execution with retry
    # ------------------------------------------------------------------

    def _wait_for_ready(
        self, timeout: float = 15.0, poll: float = 0.25, min_wait: float = 0.3,
    ) -> None:
        """Poll game state until actions are available (player can act).

        The game always responds 200 to GET /state, so we check whether
        available_actions is non-empty to know the game is ready for input.
        min_wait gives animations time to start before we begin polling.
        """
        time.sleep(min_wait)
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            try:
                gs = self.client.get_state()
                actions = gs.get("available_actions", [])
                screen = gs.get("screen", "")
                # Ready if: player has actions, or we left combat, or game over
                if actions or screen == "GAME_OVER":
                    return
                time.sleep(poll)
            except Exception:
                return
        # Timeout — proceed anyway

    def _execute_with_retry(
        self,
        action: str,
        *,
        card_index: int | None = None,
        target_index: int | None = None,
        option_index: int | None = None,
        retries: int = 10,
        delay: float = 0.3,
    ) -> dict:
        """Execute a game action, retrying on retriable 409 errors.

        The game mod returns 409 for both "action not available in current
        state" (retriable — game is animating) and permanent errors like
        "invalid_target" or "card cannot be played" (not retriable).
        """
        for attempt in range(retries + 1):
            try:
                return self.client.execute_action(
                    action,
                    card_index=card_index,
                    target_index=target_index,
                    option_index=option_index,
                )
            except ConnectionError as e:
                err_str = str(e)
                if "409" not in err_str:
                    raise
                # Permanent errors — don't retry
                if any(kw in err_str for kw in (
                    "invalid_target", "cannot be played", "out of range",
                    "is locked", "out of stock", "not supported",
                )):
                    self._log_action(
                        f"  [yellow]rejected: {err_str[:120]}[/yellow]"
                    )
                    if self.logger:
                        self.logger._emit({
                            "type": "action_rejected",
                            "action": action,
                            "card_index": card_index,
                            "target_index": target_index,
                            "option_index": option_index,
                            "error": err_str[:200],
                            "permanent": True,
                        })
                    return {}
                # Retriable (action not available — likely animating)
                if attempt < retries:
                    wait = min(delay * (1.5 ** attempt), 2.0)
                    time.sleep(wait)
                    continue
                if self.logger:
                    self.logger._emit({
                        "type": "action_rejected",
                        "action": action,
                        "card_index": card_index,
                        "target_index": target_index,
                        "option_index": option_index,
                        "error": err_str[:200],
                        "permanent": False,
                        "retries_exhausted": True,
                    })
                self._log_action(
                    f"  [yellow]skipped (game busy after {retries} retries)[/yellow]"
                )
                return {}


def _load_env_from_mcp_json() -> None:
    """Load env vars from .mcp.json if present."""
    candidate = Path(__file__).resolve().parents[3] / ".mcp.json"
    if not candidate.exists():
        candidate = Path.cwd() / ".mcp.json"
    if not candidate.exists():
        return
    try:
        with open(candidate, encoding="utf-8") as f:
            data = json.load(f)
        for server in data.get("mcpServers", {}).values():
            for key, value in server.get("env", {}).items():
                if key not in os.environ:
                    os.environ[key] = value
    except Exception:
        pass


def main():
    _load_env_from_mcp_json()
    parser = argparse.ArgumentParser(description="STS2 Autonomous Runner")
    parser.add_argument(
        "--step", action="store_true",
        help="Step mode: press Enter for each action",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Show decisions without executing",
    )
    parser.add_argument(
        "--poll", type=float, default=1.0,
        help="Seconds between state polls (default: 1.0)",
    )
    parser.add_argument(
        "--character", type=str, default=DEFAULT_CHARACTER,
        help=f"Character to play (default: {DEFAULT_CHARACTER})",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override advisor model (e.g. qwen3:8b, gemma3:4b)",
    )
    args = parser.parse_args()

    if args.model:
        os.environ["STS2_ADVISOR_MODEL"] = args.model

    runner = Runner(
        step_mode=args.step,
        dry_run=args.dry_run,
        poll_interval=args.poll,
        character=args.character,
    )
    runner.run()


if __name__ == "__main__":
    main()
