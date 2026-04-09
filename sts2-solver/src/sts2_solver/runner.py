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
    detect_screen_type,
    decide_boss_relic,
    decide_card_reward,
    decide_map,
    decide_rest,
    decide_shop,
)
from .game_data import strip_markup
from .bridge import state_from_mcp
from .constants import CardType
from .data_loader import load_cards
from .game_client import GameClient
from .game_data import load_game_data
from .run_logger import RunLogger
from .alphazero.encoding import build_vocabs_from_card_db, EncoderConfig
from .alphazero.network import STS2Network
from .alphazero.mcts import MCTS as AlphaZeroMCTS
from .alphazero.state_tensor import encode_state as az_encode_state
from .alphazero.self_play import (
    OPTION_REST, OPTION_SMITH, OPTION_SHOP_REMOVE, OPTION_SHOP_BUY,
    OPTION_SHOP_LEAVE, ROOM_TYPE_TO_OPTION,
)


DEFAULT_CHARACTER = "Ironclad"
MAX_LOG_LINES = 50


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

        # AlphaZero MCTS (initialized lazily after card_db is loaded)
        self._mcts: AlphaZeroMCTS | None = None
        self._mcts_vocabs = None
        self._mcts_config = None
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

        # Expected network-handled types (should NOT be "auto" or "deterministic")
        expected_network = {"card_reward", "map", "rest", "shop", "deck_select"}

        # Build summary lines
        lines = []
        warnings = []
        for st in sorted(routing):
            parts = [f"{src}={n}" for src, n in sorted(routing[st].items())]
            lines.append(f"  {st}: {', '.join(parts)}")
            # Flag if a network-expected type was resolved without the network
            if st in expected_network:
                non_net = sum(n for src, n in routing[st].items() if src != "network")
                net = routing[st].get("network", 0)
                if non_net > 0 and net == 0:
                    warnings.append(f"  ⚠ {st}: ALL decisions bypassed network ({non_net}x {', '.join(s for s in routing[st] if s != 'network')})")
                elif non_net > 0:
                    warnings.append(f"  ⚠ {st}: {non_net}/{net + non_net} bypassed network")

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

    # ------------------------------------------------------------------
    # Init & main loop
    # ------------------------------------------------------------------

    def _init_deps(self) -> None:
        self.console.print("[dim]Loading card database...[/dim]")
        self.card_db = load_cards()
        self.console.print(f"[dim]Loaded {len(self.card_db)} cards[/dim]")
        self.console.print("[dim]Loading game data...[/dim]")
        self.game_data = load_game_data()
        # Initialize AlphaZero MCTS
        self.console.print("[dim]Initializing AlphaZero MCTS...[/dim]")
        self._mcts_vocabs = build_vocabs_from_card_db(self.card_db)
        self._mcts_config = EncoderConfig()
        import torch
        network = STS2Network(self._mcts_vocabs, self._mcts_config)
        # Load latest checkpoint if available
        from pathlib import Path as _Path
        ckpt_dir = _Path(__file__).resolve().parents[3] / "alphazero_checkpoints"
        ckpts = sorted(ckpt_dir.glob("gen_*.pt"), key=lambda p: p.stat().st_mtime) if ckpt_dir.exists() else []
        self._checkpoint_name = None
        if ckpts:
            ckpt = torch.load(ckpts[-1], map_location="cpu", weights_only=True)
            saved = ckpt["model_state"]
            current = network.state_dict()
            compatible = {k: v for k, v in saved.items()
                          if k in current and v.shape == current[k].shape}
            skipped = set(saved.keys()) - set(compatible.keys())
            if any("trunk.0" in k for k in skipped):
                for k in [k for k in compatible if k.startswith("trunk.")]:
                    compatible.pop(k)
            network.load_state_dict(compatible, strict=False)
            self._checkpoint_name = ckpts[-1].name
            self.console.print(f"[dim]Loaded checkpoint: {self._checkpoint_name} ({len(compatible)}/{len(saved)} params)[/dim]")
        else:
            self.console.print("[dim]No checkpoint found — using random network[/dim]")
        self._mcts = AlphaZeroMCTS(
            network, self._mcts_vocabs, self._mcts_config,
            card_db=self.card_db, device="cpu",
        )
        self.logger.metadata = {
            "checkpoint": self._checkpoint_name or "none",
        }
        try:
            health = self.client.get_health()
            self.logger.game_version = health.get("game_version")
            self.console.print(f"[green]Connected to game v{self.logger.game_version}[/green]")
        except ConnectionError:
            self.console.print("[yellow]Game not reachable yet — will retry[/yellow]")

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
                    if self.step_mode:
                        # Drop out of Live temporarily for input
                        live.stop()
                        resp = input("[step] Enter=next, q=quit: ").strip().lower()
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

        if screen == "BUNDLE_SELECTION" and "choose_bundle" in actions:
            self._handle_bundle_selection()
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

        # If stuck on the same screen for too many ticks, force a default action
        if self._screen_repeat_count > 5 and not in_combat:
            self._log_action(
                f"[yellow]Stuck on {screen}/{screen_type} for {self._screen_repeat_count} ticks — forcing default[/yellow]"
            )
            self._screen_repeat_count = 0  # Reset to avoid infinite force loops
            if not self.dry_run:
                try:
                    if screen_type == "map" and "choose_map_node" in actions:
                        self._execute_with_retry("choose_map_node", option_index=0)
                    elif screen_type == "shop" and "close_shop_inventory" in actions:
                        self._execute_with_retry("close_shop_inventory")
                        self._shop_visited = True
                    else:
                        # First available action with option_index=0
                        self._execute_with_retry(actions[0], option_index=0)
                    self.action_count += 1
                except Exception as e:
                    self._log_action(f"  [red]Forced action failed: {e}[/red]")
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
        self._log_action("[dim]auto: choose_capstone_option 0[/dim]")
        if not self.dry_run:
            try:
                self._execute_with_retry("choose_capstone_option", option_index=0)
                time.sleep(1.0)
            except Exception:
                pass

    def _handle_bundle_selection(self) -> None:
        """Handle card pack/bundle selection screens (e.g. Neow's Scroll Boxes)."""
        gs = self.game_state
        actions = gs.get("available_actions", [])

        if "choose_bundle" in actions:
            self._log_action("[dim]auto: choose_bundle 0[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("choose_bundle", option_index=0)
                    time.sleep(1.0)
                except Exception:
                    pass
        elif "confirm_bundle" in actions:
            self._log_action("[dim]auto: confirm_bundle[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("confirm_bundle")
                    time.sleep(1.0)
                except Exception:
                    pass

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
                first_action, policy, root_value = self._mcts.search(
                    sim_state, num_simulations=200, temperature=0,
                )
                solve_ms = (time.perf_counter() - t0) * 1000
                total_states += 200
                total_solve_ms += solve_ms
                best_score = max(policy) if policy else 0
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

                if not self.dry_run:
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
                self.logger._emit({
                    "type": "mcts_play",
                    "card_idx": first_action.card_idx,
                    "target_idx": first_action.target_idx,
                    "card_name": label,
                    "hand": [c.name for c in hand],
                    "policy": [round(p, 4) for p in policy[:5]],
                    "value": round(root_value, 4),
                })
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

            # If we left combat (enemy died, screen changed), stop
            actions = gs.get("available_actions", [])
            if "play_card" not in actions:
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
                if "COMBAT" not in post_screen:
                    # Verify enemies are actually dead (not a mid-combat phase transition)
                    combat = post.get("combat") or {}
                    enemies = combat.get("enemies") or []
                    all_dead = not enemies or all(
                        e.get("current_hp", 0) <= 0 for e in enemies
                    )
                    if all_dead:
                        self._log_action("[bold green]Combat won![/bold green]")
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
        # But if the player has no potions, this is a transient state —
        # the game hasn't populated real actions yet. Skip this tick.
        if actions == ["discard_potion"]:
            potions = (run.get("potions") or [])
            has_potions = any(p.get("occupied") for p in potions if isinstance(p, dict))
            if not has_potions:
                self._log_action("  [dim]skip: discard_potion but no potions (transient)[/dim]")
                return
            self._log_action("  [dim]auto: discard_potion (slot 0)[/dim]")
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

            for item in reward_items:
                if not item.get("claimable", True):
                    continue
                if self._is_card_reward_item(item):
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

            # Only card rewards or nothing left.  Proceed safely.
            if "proceed" in actions:
                self._log_action("  [dim]auto: proceed (rewards done)[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("proceed")
                        self.action_count += 1
                    except Exception:
                        pass
            elif "collect_rewards_and_proceed" in actions:
                # Safe if card was already handled by intercept, or no card
                # reward exists.  Either way the intercept owns card decisions.
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
                net_decision = self._az_decide_combat_discard(
                    gs, sel, pick_worst=pick_worst)
                if net_decision is None:
                    raise RuntimeError(
                        f"Network combat select returned None — "
                        f"kind={kind!r} prompt={prompt!r} "
                        f"cards: {[c.get('name') for c in sel.get('cards', [])]}"
                    )
                pick_idx = net_decision.option_index
                self._log_action(f"  [blue]{net_decision.reasoning}[/blue]")
                self._track_decision("deck_select", "network")
                if self.logger:
                    self.logger.log_decision(
                        game_state=gs, screen_type="deck_select",
                        options=["select_deck_card"],
                        choice={"action": "select_deck_card",
                                "option_index": pick_idx,
                                "reasoning": net_decision.reasoning},
                        source="network",
                        network_value=net_decision.network_value,
                        head_scores=net_decision.head_scores,
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

        # Try network-based decisions first, fall back to deterministic
        _NETWORK_HANDLERS = {
            "rest": self._az_decide_rest,
            "map": self._az_decide_map,
            "shop": self._az_decide_shop,
            "card_reward": self._az_decide_card_reward,
            "event": self._az_decide_event,
        }
        _DETERMINISTIC_HANDLERS = {
            "rest": lambda: decide_rest(gs),
            "card_reward": lambda: decide_card_reward(gs, self.game_data),
            "map": lambda: decide_map(gs),
            "shop": lambda: decide_shop(gs, self.game_data),
            "boss_relic": lambda: decide_boss_relic(gs, self.game_data),
        }

        net_handler = _NETWORK_HANDLERS.get(screen_type)
        if net_handler:
            decision = net_handler(gs)
            if decision is not None:
                self._execute_deterministic(
                    gs, decision, screen_type, actions, run,
                )
                return

        # Deterministic fallback — only for screen types that have no
        # network handler (currently just boss_relic).  For types that DO
        # have a network handler, reaching here means the network crashed
        # (which now raises), so this is dead code for those types.
        handler = _DETERMINISTIC_HANDLERS.get(screen_type)
        if handler:
            if screen_type in _NETWORK_HANDLERS:
                # Should never reach here — network handler should have
                # either returned a decision or raised.
                raise RuntimeError(
                    f"Deterministic fallback reached for '{screen_type}' — "
                    f"network handler should have handled this"
                )
            decision = handler()
            self._execute_deterministic(
                gs, decision, screen_type, actions, run,
            )
            return

        # No handler for this screen type.
        raise RuntimeError(
            f"Unhandled screen type {screen_type!r} — no network or "
            f"deterministic handler registered. Available actions: {actions}"
        )

    # ------------------------------------------------------------------
    # AlphaZero network non-combat decisions
    # ------------------------------------------------------------------

    def _az_run_state_tensors(self, gs: dict) -> tuple:
        """Build encoded state tensors from live game state for non-combat decisions.

        Returns (state_tensors, hidden, hp, max_hp, gold, floor, deck_cards).
        """
        import torch
        from .deterministic_advisor import _get_deck, _gold, _floor
        from .models import PlayerState, CombatState

        run = gs.get("run") or {}
        hp = run.get("current_hp", 70)
        max_hp = run.get("max_hp", 70)
        gold = _gold(gs)
        floor = _floor(gs)

        # Build deck as Card objects
        deck_cards = []
        for raw in _get_deck(gs):
            card_id = raw.get("card_id") or raw.get("id", "")
            card = self.card_db.get(card_id)
            if not card and raw.get("upgraded"):
                card = self.card_db.get(card_id.rstrip("+") + "+")
            if card:
                deck_cards.append(card)

        # Relics from game state
        relic_ids = frozenset(
            r.get("id", r.get("relic_id", ""))
            for r in (run.get("relics") or [])
            if isinstance(r, dict)
        ) or frozenset(
            r for r in (run.get("relics") or [])
            if isinstance(r, str)
        )

        # Remaining map path from game state
        remaining_path = self._extract_remaining_path(gs, floor)

        player = PlayerState(hp=hp, max_hp=max_hp, energy=3, max_energy=3,
                             draw_pile=list(deck_cards))
        dummy = CombatState(
            player=player, enemies=[], floor=floor, gold=gold,
            relics=relic_ids, act_id=self._current_act_id,
            boss_id=self._current_boss_id, map_path=remaining_path,
        )
        st = az_encode_state(dummy, self._mcts_vocabs, self._mcts_config)

        with torch.no_grad():
            hidden = self._mcts.network.encode_state(**st)

        return st, hidden, hp, max_hp, gold, floor, deck_cards

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

    def _az_decide_combat_discard(self, gs: dict, sel: dict,
                                    pick_worst: bool = True) -> "Decision | None":
        """Use network to pick a card mid-combat (Survivor, Acrobatics, etc.).

        This is a WORKAROUND using OPTION_SHOP_REMOVE to score hand cards.
        The option head was trained for "how good is removing this card from
        the deck permanently?" — higher score = better removal target =
        LESS valuable card.

        pick_worst=True: discard/exhaust — pick the LEAST valuable card
            to throw away. High OPTION_SHOP_REMOVE score = less valuable
            = good discard target. Use MAX score.
        pick_worst=False: copy/duplicate — pick the MOST valuable card.
            Low OPTION_SHOP_REMOVE score = more valuable = good copy
            target. Use MIN score.
        """
        import torch
        from .deterministic_advisor import Decision
        from .alphazero.self_play import OPTION_SHOP_REMOVE

        if not self._mcts or not self._mcts_vocabs:
            return None

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network
            vocabs = self._mcts_vocabs

            cards = sel.get("cards", [])
            if not cards:
                return None

            # Score each card — use OPTION_SHOP_REMOVE type since we're
            # evaluating "how good is the deck without this card"
            opt_types = []
            opt_cards = []
            option_labels = []
            game_indices = []

            for card_info in cards:
                card_id = card_info.get("card_id") or card_info.get("id", "")
                name = card_info.get("name", "?")
                idx = card_info.get("index", len(opt_types))
                opt_types.append(OPTION_SHOP_REMOVE)
                opt_cards.append(vocabs.cards.get(card_id.rstrip("+")))
                option_labels.append(name)
                game_indices.append(idx)

            if not opt_types:
                return None

            with torch.no_grad():
                _, scores = network.pick_best_option(
                    hidden, opt_types, opt_cards)
                if pick_worst:
                    # High score = "removing is good" = card is less valuable
                    # = good discard target. Pick MAX.
                    best_idx = max(range(len(scores)), key=lambda i: scores[i])
                    verb = "discard"
                else:
                    # Low score = "removing is bad" = card is more valuable
                    # = good copy/duplicate target. Pick MIN.
                    best_idx = min(range(len(scores)), key=lambda i: scores[i])
                    verb = "select"

            chosen_idx = game_indices[best_idx]
            card_name = option_labels[best_idx]
            nv = network.value_head(hidden).item()

            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)}
                            for lbl, s in zip(option_labels, scores)],
            }

            return Decision("select_deck_card", chosen_idx,
                            f"Network: {verb} {card_name} (score={scores[best_idx]:.2f})",
                            network_value=nv, head_scores=hs)
        except Exception as e:
            self._log_action(f"  [dim]Network combat discard failed ({e})[/dim]")
            return None

    def _az_decide_rest(self, gs: dict) -> "Decision | None":
        """Use network to decide rest vs upgrade at a rest site."""
        import torch
        from .deterministic_advisor import Decision

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network
            vocabs = self._mcts_vocabs

            opt_types = [OPTION_REST]
            opt_cards = [0]
            rest_idx_map = [None]  # option idx → rest option index

            # Find rest/upgrade option indices from game state
            rest_data = gs.get("rest") or (gs.get("agent_view") or {}).get("rest") or {}
            options = rest_data.get("options", [])
            game_rest_idx, game_upgrade_idx = None, None
            for i, opt in enumerate(options):
                name = (opt.get("name") or opt.get("title") or opt.get("id", "")).lower()
                idx = opt.get("index", i)
                if "rest" in name or "heal" in name or "sleep" in name:
                    game_rest_idx = idx
                elif "upgrade" in name or "smith" in name:
                    game_upgrade_idx = idx

            # Build upgrade options
            upgrade_deck_indices = []
            if game_upgrade_idx is not None:
                for di, card in enumerate(deck):
                    if not card.upgraded and card.card_type not in ("Status", "Curse"):
                        up = self.card_db.get_upgraded(card.id)
                        if up:
                            opt_types.append(OPTION_SMITH)
                            opt_cards.append(vocabs.cards.get(card.id.rstrip("+")))
                            upgrade_deck_indices.append(di)

            with torch.no_grad():
                best_idx, scores = network.pick_best_option(hidden, opt_types, opt_cards)
                nv = network.value_head(hidden).item()

            # Build labeled scores for telemetry
            option_labels = ["Rest"]
            for di in upgrade_deck_indices:
                option_labels.append(f"Smith {deck[di].name}")
            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)} for lbl, s in zip(option_labels, scores)],
            }

            if best_idx == 0:
                return Decision("choose_rest_option",
                                game_rest_idx if game_rest_idx is not None else 0,
                                f"Network: rest (score={scores[0]:.2f})",
                                network_value=nv, head_scores=hs)
            else:
                card_di = upgrade_deck_indices[best_idx - 1]
                card_name = deck[card_di].name
                return Decision("choose_rest_option",
                                game_upgrade_idx if game_upgrade_idx is not None else 1,
                                f"Network: upgrade {card_name} (score={scores[best_idx]:.2f})",
                                network_value=nv, head_scores=hs)
        except Exception as e:
            self._log_action(f"  [red]Network rest FAILED: {e}[/red]")
            raise

    def _az_decide_map(self, gs: dict) -> "Decision | None":
        """Use network to score map node types and pick the best."""
        import torch
        from .deterministic_advisor import Decision

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network

            map_data = gs.get("map") or (gs.get("agent_view") or {}).get("map") or {}
            nodes = map_data.get("available_nodes") or map_data.get("nodes") or []
            if not nodes:
                return None

            opt_types = []
            opt_cards = []
            node_indices = []

            for i, node in enumerate(nodes):
                idx = node.get("index", i)
                t = (node.get("node_type") or node.get("type") or
                     node.get("icon") or node.get("symbol", "")).lower()

                # Map game node type to our option type
                if "elite" in t:
                    rt = "elite"
                elif "rest" in t:
                    rt = "rest"
                elif "shop" in t or "merchant" in t:
                    rt = "shop"
                elif "event" in t or "unknown" in t or "mystery" in t:
                    rt = "event"
                else:
                    rt = "normal"  # monster, treasure, etc.

                opt_type = ROOM_TYPE_TO_OPTION.get(rt)
                if opt_type is not None:
                    opt_types.append(opt_type)
                    opt_cards.append(0)
                    node_indices.append(idx)

            if not opt_types:
                return None

            with torch.no_grad():
                best_idx, scores = network.pick_best_option(hidden, opt_types, opt_cards)
                nv = network.value_head(hidden).item()

            # Build labeled scores for telemetry
            _OPT_NAMES = {v: k for k, v in ROOM_TYPE_TO_OPTION.items()}
            option_labels = [f"{_OPT_NAMES.get(ot, '?')} (node {ni})" for ot, ni in zip(opt_types, node_indices)]
            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)} for lbl, s in zip(option_labels, scores)],
            }

            chosen_node = node_indices[best_idx]
            return Decision("choose_map_node", chosen_node,
                            f"Network: node {chosen_node} (score={scores[best_idx]:.2f})",
                            network_value=nv, head_scores=hs)
        except Exception as e:
            self._log_action(f"  [red]Network map FAILED: {e}[/red]")
            raise

    def _az_decide_shop(self, gs: dict) -> "Decision | None":
        """Use network for one shop action (remove/buy/leave)."""
        import torch
        from .deterministic_advisor import Decision

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network
            vocabs = self._mcts_vocabs
            actions = gs.get("available_actions", [])

            opt_types = []
            opt_cards = []
            shop_actions = []  # (action_name, option_index, reasoning)

            shop = gs.get("shop") or (gs.get("agent_view") or {}).get("shop") or {}

            # Log shop snapshot on first visit (before any purchases)
            if not getattr(self, "_shop_snapshot_logged", False) and self.logger:
                run = gs.get("run") or {}
                self.logger._emit({
                    "type": "shop_snapshot",
                    "floor": run.get("floor", 0),
                    "gold": gold,
                    "cards": [
                        {"card_id": c.get("card_id", c.get("id", "")),
                         "name": c.get("name", ""),
                         "price": c.get("price", c.get("cost", 0)),
                         "rarity": c.get("rarity", "")}
                        for c in shop.get("cards", [])
                        if c.get("name")
                    ],
                    "relics": [
                        {"relic_id": r.get("relic_id", r.get("id", "")),
                         "name": r.get("name", ""),
                         "price": r.get("price", r.get("cost", 0))}
                        for r in shop.get("relics", [])
                    ],
                    "potions": [
                        {"name": p.get("name", ""),
                         "price": p.get("price", p.get("cost", 0))}
                        for p in shop.get("potions", [])
                    ],
                    "remove_cost": shop.get("remove_cost"),
                })
                self._shop_snapshot_logged = True

            # Remove card options
            if "remove_card_at_shop" in actions:
                remove_cost = shop.get("remove_cost", 75)
                if isinstance(remove_cost, int) and remove_cost <= gold:
                    for di, card in enumerate(deck):
                        if card.name in ("Strike", "Defend") and not card.upgraded:
                            opt_types.append(OPTION_SHOP_REMOVE)
                            opt_cards.append(vocabs.cards.get(card.id.rstrip("+")))
                            shop_actions.append(("remove_card_at_shop", None,
                                                 f"Remove {card.name}"))

            # Buy card options
            if "buy_card" in actions:
                cards = shop.get("cards", [])
                for i, card_info in enumerate(cards):
                    price = card_info.get("price", card_info.get("cost", 999))
                    if not isinstance(price, int) or price > gold or price <= 0:
                        continue
                    # Skip sold-out slots (empty name or missing card_id)
                    card_id = card_info.get("card_id") or card_info.get("id", "")
                    name = card_info.get("name", "")
                    if not card_id or not name:
                        continue
                    opt_types.append(OPTION_SHOP_BUY)
                    opt_cards.append(vocabs.cards.get(card_id.rstrip("+")))
                    shop_actions.append(("buy_card", i, f"Buy {name} ({price}g)"))

            # Leave option
            if "close_shop_inventory" in actions:
                opt_types.append(OPTION_SHOP_LEAVE)
                opt_cards.append(0)
                shop_actions.append(("close_shop_inventory", None, "Leave shop"))

            if not opt_types:
                return None

            with torch.no_grad():
                best_idx, scores = network.pick_best_option(hidden, opt_types, opt_cards)
                nv = network.value_head(hidden).item()

            # Build labeled scores for telemetry
            option_labels = [sa[2] for sa in shop_actions]
            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)} for lbl, s in zip(option_labels, scores)],
            }

            action_name, opt_idx, reason = shop_actions[best_idx]
            return Decision(action_name, opt_idx,
                            f"Network: {reason} (score={scores[best_idx]:.2f})",
                            network_value=nv, head_scores=hs)
        except Exception as e:
            self._log_action(f"  [red]Network shop FAILED: {e}[/red]")
            raise

    def _az_decide_card_reward(self, gs: dict) -> "Decision | None":
        """Use network option head to pick a card reward (or skip)."""
        import torch
        from .deterministic_advisor import Decision
        from .alphazero.self_play import OPTION_CARD_REWARD, OPTION_CARD_SKIP

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network
            vocabs = self._mcts_vocabs

            # Get offered cards from the game state.
            # Raw state: reward.card_options (on NCardRewardSelectionScreen)
            # Agent view: reward.cards (transformed by BuildAgentRewardPayload)
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

            # Build option types, card IDs, and card stats
            from .alphazero.encoding import card_stats_vector
            opt_types = []
            opt_cards = []
            opt_stats = []
            option_labels = []
            game_indices = []

            for card_info in rewards:
                name = card_info.get("name") or card_info.get("card_id", "?")
                card_id = (card_info.get("card_id") or name).rstrip("+")
                upgraded = card_info.get("upgraded", False)
                idx = card_info.get("index", len(opt_types))
                opt_types.append(OPTION_CARD_REWARD)
                opt_cards.append(vocabs.cards.get(card_id))
                # Get card stats for the option head
                card_def = self.card_db.get(card_id, upgraded=upgraded)
                if card_def:
                    opt_stats.append(card_stats_vector(card_def))
                else:
                    opt_stats.append([0.0] * self._mcts_config.card_stats_dim)
                option_labels.append(name)
                game_indices.append(idx)

            # Add skip option
            opt_types.append(OPTION_CARD_SKIP)
            opt_cards.append(0)
            opt_stats.append([0.0] * self._mcts_config.card_stats_dim)
            option_labels.append("Skip")
            game_indices.append(None)

            with torch.no_grad():
                best_idx, scores = network.pick_best_option(
                    hidden, opt_types, opt_cards,
                    option_card_stats=opt_stats)

            nv = network.value_head(hidden).item()
            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)} for lbl, s in zip(option_labels, scores)],
            }

            if best_idx < len(rewards):
                # Pick a card
                chosen_idx = game_indices[best_idx]
                card_name = option_labels[best_idx]
                self._card_reward_handled = True
                return Decision("choose_reward_card", chosen_idx,
                                f"Network: take {card_name} (score={scores[best_idx]:.2f})",
                                network_value=nv, head_scores=hs)
            else:
                # Skip — use skip_reward_cards (card selection screen)
                self._card_reward_handled = True
                return Decision("skip_reward_cards", None,
                                f"Network: skip (score={scores[best_idx]:.2f})",
                                network_value=nv, head_scores=hs)
        except Exception as e:
            self._log_action(f"  [red]Network card_reward FAILED: {e}[/red]")
            raise

    def _az_decide_event(self, gs: dict) -> "Decision | None":
        """Use network option head to decide between event options.

        Validates against event profiles to ensure we only handle events
        we've trained on.  Raises RuntimeError for unknown events.
        """
        import torch
        from .deterministic_advisor import Decision
        from .alphazero.self_play import categorize_event_option
        from .game_data import strip_markup
        from .simulator import _load_event_profiles

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network

            event = gs.get("event") or (gs.get("agent_view") or {}).get("event") or {}
            options = event.get("options") or []

            unlocked = [(i, o) for i, o in enumerate(options) if not o.get("locked")]
            if len(unlocked) <= 1:
                return None  # Let auto-handler deal with single/no options

            # Identify the event and look up its profile
            event_name = strip_markup(
                event.get("title") or event.get("name", ""))
            event_id = event.get("event_id") or event.get("id", "")

            profiles = _load_event_profiles()
            profile = profiles.get(event_id) or profiles.get(
                event_name.upper().replace(" ", "_").replace("?", "").strip("_"))

            # For Neow, match options against the pool by title
            if profile and profile.get("is_neow"):
                pool_by_title = {
                    strip_markup(o["title"]): o
                    for o in profile.get("neow_pool", [])
                }
            elif profile:
                pool_by_title = {
                    strip_markup(o["title"]): o
                    for o in profile.get("options", [])
                }
            else:
                pool_by_title = None

            if pool_by_title is None:
                raise RuntimeError(
                    f"No event profile for {event_id!r} / {event_name!r} — "
                    f"run build_event_profiles.py to add it"
                )

            opt_types = []
            opt_cards = []
            option_labels = []
            game_indices = []

            for i, opt in unlocked:
                desc = opt.get("description", "")
                title = strip_markup(
                    opt.get("title") or opt.get("name", ""))
                label = title or strip_markup(desc)[:40] or f"Option {i}"

                # Match against profile to get trained option_type
                profiled = pool_by_title.get(title)
                if profiled and "option_type" in profiled:
                    opt_types.append(profiled["option_type"])
                else:
                    raise RuntimeError(
                        f"Event option {title!r} not in profile for "
                        f"{event_name!r} — run build_event_profiles.py "
                        f"to update it"
                    )

                opt_cards.append(0)
                option_labels.append(label)
                game_indices.append(opt.get("index", i))

            with torch.no_grad():
                best_idx, scores = network.pick_best_option(
                    hidden, opt_types, opt_cards)
                nv = network.value_head(hidden).item()

            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)}
                            for lbl, s in zip(option_labels, scores)],
            }

            chosen_game_idx = game_indices[best_idx]
            chosen_label = option_labels[best_idx]

            return Decision(
                "choose_event_option", chosen_game_idx,
                f"Network: {chosen_label} (score={scores[best_idx]:.2f})",
                network_value=nv, head_scores=hs,
            )
        except Exception as e:
            self._log_action(f"  [red]Network event FAILED: {e}[/red]")
            raise

    def _az_decide_deck_select(self, gs: dict) -> "Decision | None":
        """Use network option head for deck card selection (add/remove/upgrade).

        For ALL operation types, the option head scores "how good is this
        action?" — pick_best_option (max score) is always correct.
        Do NOT invert scores for removal — the network was trained with
        max score = best removal target.
        """
        """Use network option head for card selection (removal/upgrade/transform)."""
        import torch
        from .deterministic_advisor import Decision
        from .alphazero.self_play import OPTION_SHOP_REMOVE, OPTION_SMITH, OPTION_CARD_REWARD, OPTION_EVENT_TRANSFORM

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network
            vocabs = self._mcts_vocabs

            sel = gs.get("selection") or {}
            kind = (sel.get("kind") or "").lower()
            cards = sel.get("cards", [])

            if not cards:
                return None

            # Use kind + prompt to determine operation type
            prompt_text = strip_markup(sel.get("prompt") or "").lower()
            is_upgrade = kind in ("deck_upgrade_select", "combat_hand_upgrade_select",
                                  "deck_enchant_select")
            is_remove_prompt = "remove" in prompt_text or "destroy" in prompt_text
            is_transform = "transform" in prompt_text
            is_add_prompt = ("add" in prompt_text
                             or ("choose" in prompt_text and not is_remove_prompt
                                 and not is_transform))
            is_add = not is_upgrade and not is_remove_prompt and not is_transform and is_add_prompt
            is_remove = not is_upgrade and not is_add

            # Build options for the unified option head
            opt_types = []
            opt_cards = []
            card_indices = []
            option_labels = []

            for card_info in cards:
                card_id = card_info.get("card_id") or card_info.get("id", "")
                name = card_info.get("name", "?")
                idx = card_info.get("index", len(opt_types))

                if is_upgrade:
                    opt_types.append(OPTION_SMITH)
                elif is_add:
                    opt_types.append(OPTION_CARD_REWARD)
                elif is_transform:
                    opt_types.append(OPTION_EVENT_TRANSFORM)
                else:
                    opt_types.append(OPTION_SHOP_REMOVE)

                opt_cards.append(vocabs.cards.get(card_id.rstrip("+")))
                card_indices.append(idx)
                option_labels.append(name)

            if not opt_types:
                return None

            with torch.no_grad():
                # For all operation types (remove, upgrade, add), the option
                # head scores "how good is this action?" — highest = best.
                # No inversion needed: the network was trained with
                # pick_best_option (max score) for OPTION_SHOP_REMOVE too.
                best_idx, scores = network.pick_best_option(
                    hidden, opt_types, opt_cards)

            chosen_idx = card_indices[best_idx]
            card_name = option_labels[best_idx]
            nv = network.value_head(hidden).item()
            if is_upgrade:
                action = "upgrade"
            elif is_add:
                action = "add"
            elif is_transform:
                action = "transform"
            else:
                action = "remove"

            hs = {
                "head": "option_eval",
                "chosen": best_idx,
                "options": [{"label": lbl, "score": round(s, 4)} for lbl, s in zip(option_labels, scores)],
            }

            return Decision("select_deck_card", chosen_idx,
                            f"Network: {action} {card_name} (score={scores[best_idx]:.2f})",
                            network_value=nv, head_scores=hs)
        except Exception as e:
            self._log_action(f"  [red]Network deck_select FAILED: {e}[/red]")
            raise

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
        """Single-select deck screen — network required."""
        decision = self._az_decide_deck_select(gs)
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

    def _az_score_deck_cards(self, gs: dict) -> "list[tuple[int, str, float]]":
        """Score all cards in a deck_select screen using the network.

        Returns list of (game_index, card_name, score) sorted best-first.
        Raises if the network is unavailable.
        """
        import torch
        from .alphazero.self_play import (
            OPTION_SHOP_REMOVE, OPTION_SMITH,
            OPTION_CARD_REWARD, OPTION_EVENT_TRANSFORM,
        )

        if not self._mcts or not self._mcts_vocabs:
            raise RuntimeError("Network required for deck_select but _mcts not initialized")

        st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
        network = self._mcts.network
        vocabs = self._mcts_vocabs

        sel = gs.get("selection") or {}
        kind = (sel.get("kind") or "").lower()
        cards = sel.get("cards", [])
        if not cards:
            raise RuntimeError("deck_select has no cards")

        prompt_text = strip_markup(sel.get("prompt") or "").lower()
        is_upgrade = kind in ("deck_upgrade_select", "combat_hand_upgrade_select",
                              "deck_enchant_select")
        is_remove_prompt = "remove" in prompt_text or "destroy" in prompt_text
        is_transform = "transform" in prompt_text
        is_add_prompt = ("add" in prompt_text
                         or ("choose" in prompt_text and not is_remove_prompt
                             and not is_transform))
        is_add = not is_upgrade and not is_remove_prompt and not is_transform and is_add_prompt

        opt_types = []
        opt_cards = []
        card_indices = []
        option_labels = []

        for card_info in cards:
            card_id = card_info.get("card_id") or card_info.get("id", "")
            name = card_info.get("name", "?")
            idx = card_info.get("index", len(opt_types))
            if is_upgrade:
                opt_types.append(OPTION_SMITH)
            elif is_add:
                opt_types.append(OPTION_CARD_REWARD)
            elif is_transform:
                opt_types.append(OPTION_EVENT_TRANSFORM)
            else:
                opt_types.append(OPTION_SHOP_REMOVE)
            opt_cards.append(vocabs.cards.get(card_id.rstrip("+")))
            card_indices.append(idx)
            option_labels.append(name)

        if not opt_types:
            raise RuntimeError("deck_select produced no scoreable options")

        with torch.no_grad():
            _, scores = network.pick_best_option(hidden, opt_types, opt_cards)

        # Return (game_index, name, score) sorted by score descending
        ranked = sorted(
            zip(card_indices, option_labels, scores),
            key=lambda x: x[2], reverse=True,
        )
        return ranked

    def _handle_multi_deck_select(self, gs: dict, cards: list, prompt_text: str) -> None:
        """Multi-select deck screen — use network scores to pick the top N cards."""
        import re
        multi_match = re.search(r"choose\s+(\d+)", prompt_text)
        num_to_pick = int(multi_match.group(1)) if multi_match else 2

        ranked = self._az_score_deck_cards(gs)
        priority = [idx for idx, _, _ in ranked]

        # Log the full ranking
        ranking_str = ", ".join(
            f"{name}={score:.3f}" for _, name, score in ranked
        )
        self._log_action(
            f"  [blue]Multi-select ({num_to_pick}/{len(cards)}): "
            f"network ranking: {ranking_str}[/blue]"
        )
        # Log each pick as a decision with head_scores
        if self.logger:
            nv = None
            try:
                import torch
                st, hidden, *_ = self._az_run_state_tensors(gs)
                nv = self._mcts.network.value_head(hidden).item()
            except Exception:
                pass
            hs = {
                "head": "option_eval",
                "chosen": None,  # filled per-pick below
                "options": [
                    {"label": name, "score": round(score, 4)}
                    for _, name, score in ranked
                ],
            }
            # Log top-N picks as a single grouped decision
            for pick_num in range(min(num_to_pick, len(ranked))):
                pick_idx, pick_name, pick_score = ranked[pick_num]
                hs_copy = {**hs, "chosen": pick_num}
                self.logger.log_decision(
                    game_state=gs,
                    screen_type="deck_select",
                    options=["select_deck_card"],
                    choice={
                        "action": "select_deck_card",
                        "option_index": pick_idx,
                        "reasoning": f"Network multi-select {pick_num+1}/{num_to_pick}: "
                                     f"{pick_name} (score={pick_score:.2f})",
                    },
                    source="network",
                    network_value=nv,
                    head_scores=hs_copy,
                )
        self._track_decision("deck_select", "network")

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
            self.logger.log_combat_end(gs, "defeat")
            self._combat_logged = False
            self.logger.log_run_end(gs, "defeat")
            result = "defeat"

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
        """Use network to decide card reward, then call resolve_rewards."""
        self.game_state = gs

        # Determine card choice: need card selection data.
        # If we're on the REWARD screen, the card options aren't visible
        # yet — resolve_rewards will open the picker internally.
        # If we're on CARD_SELECTION, we can evaluate now.
        card_index = None  # Default: skip
        reward_data = gs.get("reward") or {}
        av_reward = (gs.get("agent_view") or {}).get("reward") or {}
        card_options = (
            reward_data.get("card_options")
            or av_reward.get("cards")
            or (gs.get("selection") or {}).get("cards")
            or []
        )

        if card_options:
            # Network evaluates the card options
            decision = self._az_decide_card_reward(gs)
            if decision is not None:
                self._log_action(f"  [blue]{decision.reasoning}[/blue]")
                self._track_decision("card_reward", "network")
                if self.logger:
                    self.logger.log_decision(
                        game_state=gs, screen_type="card_reward",
                        options=gs.get("available_actions", []),
                        choice={"action": decision.action,
                                "option_index": decision.option_index,
                                "reasoning": decision.reasoning},
                        source="network",
                        network_value=decision.network_value,
                        head_scores=decision.head_scores,
                    )
                if decision.action == "choose_reward_card":
                    card_index = decision.option_index
                else:
                    card_index = None  # Skip
                    self._deck_size_after_skip = len(
                        (gs.get("run") or {}).get("deck", [])
                    )
        else:
            self._log_action("  [dim]No card options visible — resolve_rewards will handle[/dim]")

        # Single atomic call: claims gold/potion/relic + takes/skips card
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
                # Fall back to collect_rewards_and_proceed
                try:
                    self._execute_with_retry("collect_rewards_and_proceed")
                except Exception:
                    pass

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
