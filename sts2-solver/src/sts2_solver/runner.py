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

from .advisor import StrategicAdvisor
from .advisor_prompts import AUTO_ACTIONS, detect_screen_type
from .deterministic_advisor import (
    decide_boss_relic,
    decide_card_reward,
    decide_deck_select,
    decide_map,
    decide_rest,
    decide_shop,
)
from .game_data import strip_markup
from .bridge import state_from_mcp
from .data_loader import load_cards
from .game_client import GameClient
from .game_data import load_game_data
from .run_logger import RunLogger
from .solver import solve_turn, format_solution
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
    ):
        self.step_mode = step_mode
        self.dry_run = dry_run
        self.poll_interval = poll_interval
        self.character = character

        self.console = Console()
        self.client = GameClient()
        self.card_db = None
        self.game_data = None
        self.advisor = None
        self.logger = RunLogger(logs_dir=Path(logs_dir) if logs_dir else None)

        self.game_state: dict | None = None
        self.turn_count = 0
        self.action_count = 0

        # AlphaZero MCTS (initialized lazily after card_db is loaded)
        self._mcts: AlphaZeroMCTS | None = None
        self._mcts_vocabs = None
        self._mcts_config = None
        self._card_reward_handled = False  # Reset when leaving reward screen
        self._deck_select_stuck = False  # Track stuck deck_select screens
        self._stuck_since: float | None = None  # Timestamp when we got stuck
        self._shop_visited = False  # Prevent re-opening shop after closing
        self._last_floor: int | None = None  # Track floor for shop reset
        self._last_screen_key: tuple[str, str] | None = None  # (screen, screen_type)
        self._screen_repeat_count: int = 0  # Same-screen repeat counter
        self._combat_move_indices: dict[tuple[int, str], int] = {}  # Enemy move cycle tracking

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
            Layout(name="panels", ratio=3),
            Layout(name="log", ratio=2),
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

        # Log panel
        log_text = "\n".join(self._log) if self._log else "[dim]Waiting...[/dim]"
        layout["log"].update(
            Panel(Text.from_markup(log_text), title="Action Log", border_style="green")
        )

        return layout

    def _log_action(self, msg: str) -> None:
        self._log.append(msg)

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
        self.advisor = StrategicAdvisor(
            self.game_data, self.client, logger=self.logger
        )

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
            "advisor_model": self.advisor.model,
            "advisor_local": self.advisor.is_local,
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
                self._update_dashboard()

    def _update_dashboard(self) -> None:
        """Rebuild data.json and deploy to Vercel after a run."""
        script = Path(__file__).resolve().parents[3] / "dashboard" / "update_data.py"
        if not script.exists():
            return
        try:
            import subprocess, sys
            subprocess.run(
                [sys.executable, str(script), "--deploy"],
                timeout=60,
                capture_output=True,
            )
        except Exception:
            pass

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

        screen = self.game_state.get("screen", "")

        # Reset card reward tracking when we leave the reward screen
        if screen not in ("REWARD", "CARD_SELECTION"):
            self._card_reward_handled = False

        # Reset shop visit flag when the floor changes (not when screen changes)
        run = self.game_state.get("run") or {}
        current_floor = run.get("floor")
        if current_floor is not None and current_floor != self._last_floor:
            self._shop_visited = False
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

        if turn == 1 or (isinstance(turn, int) and turn <= 1):
            # Wait briefly for enemy intents to be revealed by the game
            time.sleep(0.5)
            gs = self.client.get_state()
            self.game_state = gs
            combat = gs.get("combat") or {}
            player = combat.get("player") or {}
            enemies = combat.get("enemies") or []

            self.logger.log_combat_start(gs)
            self._combat_move_indices = {}

        # Potions are now handled by MCTS as part of the action space —
        # the network decides when to use potions during the play loop.

        # Snapshot the pre-play state for combat logging
        turn_start_gs = gs

        # Solve-one-play-re-solve loop: play one card at a time from fresh
        # game state. This accounts for relic triggers, energy generation,
        # card cost changes, and other effects the simulator doesn't model.
        from .bridge import action_to_mcp

        cards_played: list[str] = []
        targets_chosen: list[int | None] = []
        total_states = 0
        total_solve_ms = 0.0
        best_score = 0.0
        max_cards = 12  # safety cap to prevent infinite loops

        while len(cards_played) < max_cards:
            # Build combat state and run MCTS
            try:
                sim_state = state_from_mcp(gs, self.card_db,
                                          move_indices=self._combat_move_indices)
                hand = list(sim_state.player.hand)
                t0 = time.perf_counter()
                first_action, policy = self._mcts.search(
                    sim_state, num_simulations=200, temperature=0,
                )
                solve_ms = (time.perf_counter() - t0) * 1000
                total_states += 200
                total_solve_ms += solve_ms
                best_score = max(policy) if policy else 0
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
                pot_name = "potion"
                potions_raw = (gs.get("run") or {}).get("potions", [])
                for p in potions_raw:
                    if p.get("index") == first_action.potion_idx:
                        pot_name = p.get("name", "potion")
                        break
                label = f"Use {pot_name} (slot {first_action.potion_idx})"
                cards_played.append(label)
                targets_chosen.append(first_action.target_idx)

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
                intents = e_raw.get("intents", [])
                it, idmg, ihits = None, None, 1
                for intent in intents:
                    itype = intent.get("intent_type", "")
                    if itype == "Attack":
                        it = "Attack"
                        idmg = intent.get("damage")
                        ihits = intent.get("hits", 1)
                    elif itype in ("Defend", "Buff", "Debuff", "StatusCard"):
                        it = it or itype
                idx = _match_move_index(eid, it, idmg, ihits)
                if idx is not None:
                    self._combat_move_indices[key] = idx

        # End turn if we're still in combat
        if not self.dry_run and "end_turn" in gs.get("available_actions", []):
            self._wait_for_ready()
            try:
                self._execute_with_retry("end_turn")
                self._log_action("  [green]>[/green] End Turn")
                self.action_count += 1
            except Exception as e:
                self._log_action(f"  [red]X End Turn: {e}[/red]")

        # Log the full turn (pass pre-play state for combat snapshot)
        self.logger.log_combat_turn(
            cards_played=cards_played,
            targets_chosen=targets_chosen,
            score=best_score,
            states_evaluated=total_states,
            solve_ms=total_solve_ms,
            game_state=turn_start_gs,
        )

        self.turn_count += 1

        # Check for combat end — only log "win" if all enemies are dead,
        # not just because the screen changed (boss phase transitions leave
        # combat temporarily for card selection screens).
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
        except Exception:
            pass

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
        if actions == ["discard_potion"]:
            self._log_action("  [dim]auto: discard_potion (slot 0)[/dim]")
            if not self.dry_run:
                try:
                    self._execute_with_retry("discard_potion", option_index=0)
                    self.action_count += 1
                except Exception as e:
                    self._log_action(f"  [red]Failed to discard potion: {e}[/red]")
            return

        # Reward screen: collect_rewards_and_proceed auto-picks the first
        # card reward — NEVER use it when an unhandled card choice exists.
        # Instead, claim the card reward item to open the selection screen,
        # then let the advisor choose or skip.
        #
        # IMPORTANT: collect_rewards_and_proceed also auto-claims skipped card
        # rewards. After a skip, we must claim non-card rewards individually
        # first, then proceed only when no card rewards remain claimable.
        if "collect_rewards_and_proceed" in actions and screen_type != "card_reward":
            reward = gs.get("reward") or {}
            if not reward:
                reward = (gs.get("agent_view") or {}).get("reward") or {}

            # Also check agent_view reward for pending_card_choice
            agent_reward = (gs.get("agent_view") or {}).get("reward") or {}

            has_card_choice = (
                reward.get("pending_card_choice")
                or agent_reward.get("pending_card_choice")
                or "choose_reward_card" in actions
                or "skip_reward_cards" in actions
            )

            # Check reward items for card-type rewards.
            # Raw state: reward_type="Card"; agent_view: line="card: ...".
            reward_items = reward.get("rewards") or []
            if not reward_items:
                reward_items = agent_reward.get("rewards") or []
            has_card_reward_item = any(
                self._is_card_reward_item(item)
                for item in reward_items
                if item.get("claimable", True)
            )

            # If reward data is empty but we just arrived at the reward screen,
            # wait for the data to populate before auto-proceeding.
            if not reward_items and not has_card_choice and "claim_reward" in actions:
                return  # Let next tick re-check once reward data is populated

            # Debug: log reward detection state
            if "claim_reward" in actions:
                self._log_action(
                    f"  [dim]reward check: items={len(reward_items)} "
                    f"card_choice={has_card_choice} card_item={has_card_reward_item}[/dim]"
                )

            # If we already handled the card choice this reward screen,
            # claim non-card rewards individually to avoid collect_rewards_and_proceed
            # which auto-grabs the first card (even after skip_reward_cards).
            if self._card_reward_handled:
                self._card_reward_handled = False
                if "claim_reward" in actions:
                    # Claim first non-card reward item
                    for item in reward_items:
                        if not self._is_card_reward_item(item) and item.get("claimable", True):
                            idx = item.get("index", item.get("i"))
                            if idx is not None:
                                self._log_action(f"  [dim]auto: claim_reward({idx}) — non-card[/dim]")
                                if not self.dry_run:
                                    try:
                                        self._execute_with_retry("claim_reward", option_index=idx)
                                        self.action_count += 1
                                    except Exception:
                                        pass
                                return
                # No non-card rewards left, or no claim_reward action.
                # Try proceed first (doesn't auto-claim), fall back to
                # collect_rewards_and_proceed only if proceed isn't available.
                if "proceed" in actions:
                    self._log_action("  [dim]auto: proceed (post-skip)[/dim]")
                    if not self.dry_run:
                        try:
                            self._execute_with_retry("proceed")
                            self.action_count += 1
                        except Exception:
                            pass
                elif "collect_rewards_and_proceed" in actions:
                    self._log_action("  [dim]auto: collect_rewards_and_proceed (post-skip)[/dim]")
                    if not self.dry_run:
                        try:
                            self._execute_with_retry("collect_rewards_and_proceed")
                            self.action_count += 1
                        except Exception:
                            pass
                return

            if has_card_choice or has_card_reward_item:
                if "choose_reward_card" in actions or "skip_reward_cards" in actions:
                    # Card selection screen is open — let advisor handle it
                    screen_type = "card_reward"
                    # Fall through to LLM decision below
                elif has_card_reward_item and "claim_reward" in actions:
                    # Open the card selection screen by claiming the card reward
                    card_reward_idx = None
                    for item in reward_items:
                        if self._is_card_reward_item(item) and item.get("claimable", True):
                            card_reward_idx = item.get("index", item.get("i"))
                            break
                    if card_reward_idx is not None:
                        self._log_action(f"  [cyan]Opening card reward (index {card_reward_idx})...[/cyan]")
                        if not self.dry_run:
                            try:
                                self._execute_with_retry("claim_reward", option_index=card_reward_idx)
                                time.sleep(1.0)  # Wait for card data to populate
                            except Exception as e:
                                self._log_action(f"  [red]Failed to open card reward: {e}[/red]")
                    return
                else:
                    # Not ready yet — return and let next tick handle it
                    return
            else:
                # No card choice pending — safe to auto-proceed
                self._log_action("  [dim]auto: collect_rewards_and_proceed[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("collect_rewards_and_proceed")
                        self.action_count += 1
                    except Exception as e:
                        self._log_action(f"  [red]Auto-action failed: {e}[/red]")
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
                self._log_action("  [dim]auto: select_deck_card(0) — dismiss card preview overlay[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=0)
                    except Exception:
                        pass
                return

        # Multi-select deck screens (e.g. "Choose 2 cards to Add/Remove"):
        # select_deck_card: check if this is a real decision or an
        # informational overlay (e.g. Havoc showing "Draw 3 cards")
        if screen_type == "deck_select":
            # If we already tried and failed on this screen, skip it
            if self._deck_select_stuck:
                self._log_action("  [dim]Skipping stuck deck_select screen[/dim]")
                return

            sel = gs.get("selection") or {}
            prompt = strip_markup(sel.get("prompt") or "").lower()

            # "Confirm" screens (e.g. Armaments "Confirm Card to Upgrade"):
            # A card was already selected — just re-select index 0 to confirm.
            if "confirm" in prompt:
                self._log_action(f"  [dim]auto: select_deck_card(0) — confirm[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=0)
                    except Exception:
                        pass
                return

            # Mid-combat card selections (Havoc "put on top of Draw Pile",
            # "Choose a card to Exhaust", etc.): pick the first non-essential
            # card quickly instead of calling the LLM.
            _COMBAT_SELECT_KW = (
                "draw pile", "exhaust", "discard pile", "put on top",
            )
            is_combat_select = any(kw in prompt for kw in _COMBAT_SELECT_KW)

            # Decision keywords that need advisor input (non-combat only)
            is_decision = not is_combat_select and any(kw in prompt for kw in (
                "choose", "remove", "upgrade", "transform", "add", "select",
            ))

            if is_combat_select:
                # Quick deterministic pick: avoid key card, prefer Strikes/Defends
                from .config import CHARACTER_CONFIG, detect_character
                _char = detect_character(gs)
                _key = CHARACTER_CONFIG.get(_char, {}).get("key_card", "Bash").lower()
                cards = sel.get("cards", [])
                pick_idx = 0
                for card in cards:
                    name = (card.get("name") or "").lower()
                    if _key not in name:
                        pick_idx = card.get("index", card.get("i", 0))
                        break
                self._log_action(f"  [dim]auto: select_deck_card({pick_idx}) — combat select[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=pick_idx)
                    except Exception:
                        pass
                return
            elif is_decision:
                self._handle_deck_select(gs)
            else:
                # Informational overlay — auto-select first card to dismiss
                self._log_action(f"  [dim]auto: select_deck_card (overlay)[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("select_deck_card", option_index=0)
                    except Exception:
                        pass
            return

        # For finished events with only a "Proceed" option, auto-handle
        if screen_type == "event" and "choose_event_option" in actions:
            event = gs.get("event") or {}
            options = event.get("options") or []
            if event.get("finished") or (
                len(options) == 1 and options[0].get("proceed")
            ):
                self._log_action("  [dim]auto: choose_event_option(0) — proceed[/dim]")
                if not self.dry_run:
                    try:
                        self._execute_with_retry("choose_event_option", option_index=0)
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

        handler = _DETERMINISTIC_HANDLERS.get(screen_type)
        if handler:
            decision = handler()
            self._execute_deterministic(
                gs, decision, screen_type, actions, run,
            )
            return

        # Events + generic: LLM-based decision (only remaining LLM usage)
        try:
            result_str = self.advisor.advise(gs, execute=not self.dry_run)
        except Exception as e:
            self._log_action(f"[red]Advisor error: {e}[/red]")
            return

        # If the advisor recommended an invalid/failed action, fall back to a safe default
        if "not available" in result_str or "FAILED" in result_str:
            self._log_action(f"  [yellow]Invalid action — falling back[/yellow]")
            _FALLBACKS = [
                ("choose_event_option", 0),
                ("proceed", None),
                ("confirm_modal", None),
                ("dismiss_modal", None),
            ]
            for fb_action, fb_idx in _FALLBACKS:
                if fb_action not in actions:
                    continue
                if not self.dry_run:
                    try:
                        self._execute_with_retry(fb_action, option_index=fb_idx)
                    except Exception:
                        pass
                self._log_action(f"  [dim]auto: {fb_action}({fb_idx}) (fallback)[/dim]")
                self.action_count += 1
                return
            return

        # Update advisor panel
        self._advisor_text = (
            f"[bold]{screen_type.upper()}[/bold] | "
            f"Floor {run.get('floor', '?')} | "
            f"HP {run.get('current_hp', '?')}/{run.get('max_hp', '?')}\n\n"
            f"{result_str}"
        )

        lines = result_str.split("\n")
        decision_line = next((l for l in lines if l.startswith("Decision:")), lines[0] if lines else "?")
        self._log_action(f"  [blue]{decision_line}[/blue]")
        self.action_count += 1
        self._refresh()

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

        player = PlayerState(hp=hp, max_hp=max_hp, energy=3, max_energy=3,
                             draw_pile=list(deck_cards))
        dummy = CombatState(player=player, enemies=[], floor=floor, gold=gold)
        st = az_encode_state(dummy, self._mcts_vocabs, self._mcts_config)

        with torch.no_grad():
            hidden = self._mcts.network.encode_state(**st)

        return st, hidden, hp, max_hp, gold, floor, deck_cards

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

            if best_idx == 0:
                return Decision("choose_rest_option",
                                game_rest_idx if game_rest_idx is not None else 0,
                                f"Network: rest (score={scores[0]:.2f})")
            else:
                card_di = upgrade_deck_indices[best_idx - 1]
                card_name = deck[card_di].name
                return Decision("choose_rest_option",
                                game_upgrade_idx if game_upgrade_idx is not None else 1,
                                f"Network: upgrade {card_name} (score={scores[best_idx]:.2f})")
        except Exception as e:
            self._log_action(f"  [dim]Network rest failed ({e}), falling back[/dim]")
            return None

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

            chosen_node = node_indices[best_idx]
            return Decision("choose_map_node", chosen_node,
                            f"Network: node {chosen_node} (score={scores[best_idx]:.2f})")
        except Exception as e:
            self._log_action(f"  [dim]Network map failed ({e}), falling back[/dim]")
            return None

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
                    if not isinstance(price, int) or price > gold:
                        continue
                    card_id = card_info.get("card_id") or card_info.get("id", "")
                    opt_types.append(OPTION_SHOP_BUY)
                    opt_cards.append(vocabs.cards.get(card_id.rstrip("+").rstrip("+")))
                    name = card_info.get("name", card_id)
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

            action_name, opt_idx, reason = shop_actions[best_idx]
            return Decision(action_name, opt_idx,
                            f"Network: {reason} (score={scores[best_idx]:.2f})")
        except Exception as e:
            self._log_action(f"  [dim]Network shop failed ({e}), falling back[/dim]")
            return None

    def _az_decide_deck_select(self, gs: dict) -> "Decision | None":
        """Use network deck_eval_head for card removal/upgrade/transform."""
        import torch
        from .deterministic_advisor import Decision

        try:
            st, hidden, hp, max_hp, gold, floor, deck = self._az_run_state_tensors(gs)
            network = self._mcts.network
            vocabs = self._mcts_vocabs

            sel = gs.get("selection") or {}
            prompt = (sel.get("prompt") or "").lower()
            cards = sel.get("cards", [])

            if not cards:
                return None

            is_remove = "remove" in prompt or "transform" in prompt
            is_upgrade = "upgrade" in prompt

            # Build card IDs for evaluation
            card_ids = []
            card_indices = []  # game option indices
            for card_info in cards:
                card_id = card_info.get("card_id") or card_info.get("id", "")
                idx = card_info.get("index", len(card_ids))

                if is_upgrade:
                    # Score the upgraded version
                    up_id = card_id.rstrip("+") + "+"
                    card_ids.append(vocabs.cards.get(up_id.rstrip("+")))
                else:
                    card_ids.append(vocabs.cards.get(card_id.rstrip("+")))
                card_indices.append(idx)

            if not card_ids:
                return None

            with torch.no_grad():
                ids_t = torch.tensor([card_ids], dtype=torch.long)
                scores = network.evaluate_deck_change(hidden, ids_t)
                scores_list = scores[0].tolist()

            if is_remove:
                # Remove: pick lowest-scored card
                best = min(range(len(scores_list)), key=lambda i: scores_list[i])
            else:
                # Upgrade: pick highest-scored card
                best = max(range(len(scores_list)), key=lambda i: scores_list[i])

            chosen_idx = card_indices[best]
            card_name = cards[best].get("name", "?")
            action = "remove" if is_remove else "upgrade"
            return Decision("select_deck_card", chosen_idx,
                            f"Network: {action} {card_name} (score={scores_list[best]:.2f})")
        except Exception as e:
            self._log_action(f"  [dim]Network deck_select failed ({e}), falling back[/dim]")
            return None

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
            }
            fb = _FALLBACKS.get(screen_type)
            if fb and fb[0] in actions:
                decision = Decision(fb[0], fb[1], "fallback")
            else:
                return

        # Log the decision
        self._log_action(
            f"  [blue]Decision: {decision.action}"
            f"{f' ({decision.option_index})' if decision.option_index is not None else ''}"
            f" — {decision.reasoning}[/blue]"
        )

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
                source="deterministic",
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
        if screen_type == "shop" and decision.action == "close_shop_inventory":
            self._shop_visited = True

        # Update advisor panel
        self._advisor_text = (
            f"[bold]{screen_type.upper()}[/bold] | "
            f"Floor {run.get('floor', '?')} | "
            f"HP {run.get('current_hp', '?')}/{run.get('max_hp', '?')}\n\n"
            f"[green]\\[deterministic][/green] {decision.action}"
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
        """Single-select deck screen — try network, fall back to deterministic."""
        decision = self._az_decide_deck_select(gs)
        if decision is None:
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

    def _handle_multi_deck_select(self, gs: dict, cards: list, prompt_text: str) -> None:
        """Multi-select deck screen — pick deterministically.

        For remove: Strikes first, then Defends, never key card.
        For other multi-selects: pick sequentially from index 0.
        """
        import re
        multi_match = re.search(r"choose\s+(\d+)", prompt_text)
        num_to_pick = int(multi_match.group(1)) if multi_match else 2

        is_remove = "remove" in prompt_text
        is_upgrade = "upgrade" in prompt_text

        # Build priority order for indices
        if is_remove:
            # Remove Strikes first, then Defends, then others, NEVER key card
            from .config import CHARACTER_CONFIG, detect_character
            _char = detect_character(gs)
            _key = CHARACTER_CONFIG.get(_char, {}).get("key_card", "Bash").lower()
            priority = []
            for card in cards:
                name = (card.get("name") or "").lower()
                idx = card.get("index", 0)
                if _key in name:
                    continue  # Never remove key card
                if "strike" in name:
                    priority.insert(0, idx)  # Strikes first
                elif "defend" in name:
                    priority.append(idx)  # Defends after Strikes
                else:
                    priority.append(idx)  # Others last
            self._log_action(f"  [cyan]Multi-remove: picking {num_to_pick} from priority {priority[:num_to_pick]}[/cyan]")
        else:
            # For upgrade/other: just pick sequentially
            priority = [card.get("index", i) for i, card in enumerate(cards)]

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
            self.logger.log_run_end(gs, "victory")
        else:
            self._log_action(
                f"[bold red]DEFEAT[/bold red] Floor {floor} | HP {hp}"
            )
            self.logger.log_combat_end(gs, "defeat")
            self.logger.log_run_end(gs, "defeat")

        self._log_action(
            f"Turns: {self.turn_count} | Actions: {self.action_count}"
        )
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
                    return {}
                # Retriable (action not available — likely animating)
                if attempt < retries:
                    wait = min(delay * (1.5 ** attempt), 2.0)
                    time.sleep(wait)
                    continue
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
