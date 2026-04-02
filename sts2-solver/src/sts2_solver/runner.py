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
from .bridge import state_from_mcp
from .data_loader import load_cards
from .game_client import GameClient
from .game_data import load_game_data
from .run_logger import RunLogger
from .solver import solve_turn, format_solution


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
        self.logger = RunLogger()

        self.game_state: dict | None = None
        self.turn_count = 0
        self.action_count = 0

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

        if not actions:
            return False

        self.logger.ensure_run(self.game_state)

        screen = self.game_state.get("screen", "")
        in_combat = (
            "play_card" in actions
            or ("end_turn" in actions and "COMBAT" in screen.upper())
        )

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

    # ------------------------------------------------------------------
    # Combat
    # ------------------------------------------------------------------

    def _should_use_potion(self, gs: dict) -> tuple[int, int | None] | None:
        """Decide whether to use a potion this turn. Returns (slot, target) or None.

        Simple heuristics:
        - Use damage/attack potions if an enemy is close to lethal
        - Use block/defense potions if incoming damage would kill us
        - Use strength/buff potions on turn 1-2 for scaling
        - Otherwise save them
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

        # Find usable potions
        for pot in potions:
            if not pot.get("occupied") or not pot.get("can_use"):
                continue
            slot = pot.get("index", 0)
            name = (pot.get("name") or "").lower()
            desc = (pot.get("description") or "").lower()
            needs_target = pot.get("requires_target", False)

            # First alive enemy as default target
            first_alive = None
            for e in enemies:
                if e.get("current_hp", 0) > 0:
                    first_alive = e.get("index", 0)
                    break

            target = first_alive if needs_target else None

            # Use healing potions if HP < 40%
            if hp_pct < 0.4 and any(kw in name or kw in desc for kw in ("heal", "blood", "fairy", "fruit")):
                return (slot, target)

            # Use block potions if we'd die this turn
            if would_die and any(kw in name or kw in desc for kw in ("block", "ghost", "shield")):
                return (slot, target)

            # Use damage potions if we'd die (offense as defense — kill them first)
            if would_die and any(kw in name or kw in desc for kw in (
                "fire", "attack", "explosive", "damage", "poison", "lightning",
            )):
                return (slot, target)

            # Use strength/buff potions if HP is okay and it's early in combat
            turn = gs.get("turn", 0)
            if turn <= 2 and hp_pct > 0.5 and any(kw in name or kw in desc for kw in (
                "strength", "flex", "dexterity", "energy", "speed",
            )):
                return (slot, target)

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
            self.logger.log_combat_start(gs)

        # Check if we should use a potion before solving
        potion_use = self._should_use_potion(gs)
        if potion_use and not self.dry_run:
            slot, target = potion_use
            pot_name = "potion"
            for p in (gs.get("run") or {}).get("potions", []):
                if p.get("index") == slot:
                    pot_name = p.get("name", "potion")
                    break
            self._log_action(f"  [magenta]Using {pot_name} (slot {slot})[/magenta]")
            try:
                self._execute_with_retry(
                    "use_potion", option_index=slot, target_index=target,
                )
                time.sleep(0.5)
                # Re-fetch state after potion use
                gs = self.client.get_state()
                self.game_state = gs
                combat = gs.get("combat") or {}
                player = combat.get("player") or {}
                enemies = combat.get("enemies") or []
                # If potion use changed available actions (e.g. no more play_card), bail
                if "play_card" not in gs.get("available_actions", []):
                    return
            except Exception as e:
                self._log_action(f"  [yellow]Potion use failed: {e}[/yellow]")

        # Solve
        try:
            sim_state = state_from_mcp(gs, self.card_db)
            original_hand = list(sim_state.player.hand)  # snapshot before solve
            t0 = time.perf_counter()
            result = solve_turn(sim_state, card_db=self.card_db)
            solve_ms = (time.perf_counter() - t0) * 1000
        except Exception as e:
            self._log_action(f"[red]Solver error: {e}[/red]")
            return

        # Restore hand for format_solution (solve_turn may have mutated it)
        sim_state.player.hand = original_hand

        # Update solver panel
        solution_str = format_solution(result, sim_state)
        hand_str = ", ".join(c.name for c in original_hand)
        self._solver_text = (
            f"[bold]Turn {turn}[/bold] | "
            f"HP {player.get('current_hp', '?')}/{player.get('max_hp', '?')} | "
            f"Energy {player.get('energy', '?')}\n"
            f"Hand: {hand_str}\n"
            f"vs: {enemy_str}\n\n"
            f"{solution_str}"
        )
        self._refresh()

        # Log
        hand = list(sim_state.player.hand)
        cards_played = []
        for a in result.actions:
            if a.card_idx is not None and a.card_idx < len(hand):
                cards_played.append(hand[a.card_idx].name)
                hand.pop(a.card_idx)
        self.logger.log_combat_turn(
            cards_played=cards_played,
            score=result.score,
            states_evaluated=result.states_evaluated,
            solve_ms=solve_ms,
        )

        if self.dry_run:
            self._log_action(f"  [dim]\\[dry-run] Would play: {', '.join(cards_played)}[/dim]")
            return

        # Execute
        from .bridge import actions_to_mcp_sequence
        mcp_actions = actions_to_mcp_sequence(result.actions)
        exec_hand = list(original_hand)

        for i, (solver_action, mcp_action) in enumerate(zip(result.actions, mcp_actions)):
            if solver_action.action_type == "end_turn":
                label = "End Turn"
            elif solver_action.card_idx is not None and solver_action.card_idx < len(exec_hand):
                card = exec_hand[solver_action.card_idx]
                target = (
                    f" -> enemy {solver_action.target_idx}"
                    if solver_action.target_idx is not None else ""
                )
                label = f"{card.name}{target}"
                exec_hand.pop(solver_action.card_idx)
            else:
                label = f"card_idx={solver_action.card_idx}"

            # Wait for game to be ready before sending each action
            # (skip for the first action — game is already ready from solve)
            if i > 0:
                self._wait_for_ready()

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

        # collect_rewards_and_proceed: auto only when no card reward pending.
        # The reward screen loads in stages — card choices may not be in the
        # first state poll. Re-check after a delay to avoid skipping rewards.
        if "collect_rewards_and_proceed" in actions and screen_type != "card_reward":
            # Double-check: wait briefly and re-poll to see if card reward appears
            if not self.dry_run:
                time.sleep(1.0)
                try:
                    fresh = self.client.get_state()
                    fresh_actions = fresh.get("available_actions", [])
                    if "choose_reward_card" in fresh_actions:
                        # Card reward appeared — handle it via advisor instead
                        self.game_state = fresh
                        gs = fresh
                        actions = fresh_actions
                        screen_type = "card_reward"
                        # Fall through to LLM-based decision below
                    else:
                        self._log_action("  [dim]auto: collect_rewards_and_proceed[/dim]")
                        self._execute_with_retry("collect_rewards_and_proceed")
                        self.action_count += 1
                        self.logger.log_decision(
                            game_state=gs, screen_type="auto", options=actions,
                            choice={"action": "collect_rewards_and_proceed", "option_index": None},
                            source="auto",
                        )
                        return
                except Exception as e:
                    self._log_action(f"  [red]Auto-action failed: {e}[/red]")
                    return
            else:
                self._log_action("  [dim]auto: collect_rewards_and_proceed[/dim]")
                self.logger.log_decision(
                    game_state=gs, screen_type="auto", options=actions,
                    choice={"action": "collect_rewards_and_proceed", "option_index": None},
                    source="auto",
                )
                return

        # Auto-actions
        if screen_type == "auto":
            for action in actions:
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

        # Multi-select deck screens (e.g. "Choose 2 cards to Add/Remove"):
        # select_deck_card toggles a card on/off. We must track which cards
        # are already selected and loop until the screen changes or
        # confirm_selection appears.
        if screen_type == "deck_select":
            self._handle_deck_select(gs)
            return

        # LLM-based decision
        try:
            result_str = self.advisor.advise(gs, execute=not self.dry_run)
        except Exception as e:
            self._log_action(f"[red]Advisor error: {e}[/red]")
            return

        # Update advisor panel
        self._advisor_text = (
            f"[bold]{screen_type.upper()}[/bold] | "
            f"Floor {run.get('floor', '?')} | "
            f"HP {run.get('current_hp', '?')}/{run.get('max_hp', '?')}\n\n"
            f"{result_str}"
        )

        # Extract the decision line for the log
        lines = result_str.split("\n")
        decision_line = next((l for l in lines if l.startswith("Decision:")), lines[0] if lines else "?")
        self._log_action(f"  [blue]{decision_line}[/blue]")
        self.action_count += 1
        self._refresh()

    # ------------------------------------------------------------------
    # Multi-select deck screens
    # ------------------------------------------------------------------

    def _handle_deck_select(self, gs: dict) -> None:
        """Handle deck card selection screens (add, remove, upgrade, transform).

        select_deck_card toggles cards on/off. After each pick we re-poll
        and check: did the screen change? Did confirm_selection appear?
        Do we need more picks?
        """
        max_picks = 10  # safety limit
        consecutive_failures = 0

        for pick in range(max_picks):
            # Ask the advisor
            try:
                result_str = self.advisor.advise(gs, execute=not self.dry_run)
            except Exception as e:
                self._log_action(f"[red]Advisor error: {e}[/red]")
                return

            # Check if the advisor actually executed successfully
            executed_ok = "-> OK" in result_str

            # Extract what was picked
            lines = result_str.split("\n")
            decision_line = next(
                (l for l in lines if l.startswith("Decision:")),
                lines[0] if lines else "?",
            )
            self._log_action(f"  [blue]{decision_line}[/blue]")
            self.action_count += 1
            self._refresh()

            if not executed_ok:
                consecutive_failures += 1
                self._log_action(f"  [yellow]Advisor did not execute (attempt {consecutive_failures})[/yellow]")
                if consecutive_failures >= 3:
                    # Fall back: just pick the first card
                    self._log_action("  [yellow]Falling back to first card[/yellow]")
                    if not self.dry_run:
                        try:
                            self._execute_with_retry("select_deck_card", option_index=0)
                        except Exception:
                            pass
                    # Continue to re-poll below
                else:
                    time.sleep(0.5)
                    continue
            else:
                consecutive_failures = 0

            # Wait and re-poll
            time.sleep(0.5)
            try:
                gs = self.client.get_state()
            except Exception:
                return
            self.game_state = gs

            screen = gs.get("screen", "")
            actions = gs.get("available_actions", [])

            # confirm_selection available — auto-confirm and done
            if "confirm_selection" in actions:
                if not self.dry_run:
                    try:
                        self._execute_with_retry("confirm_selection")
                    except Exception:
                        pass
                self._log_action("  [dim]auto: confirm_selection[/dim]")
                return

            # Screen changed away from card selection — done
            if "select_deck_card" not in actions:
                return

            # Still on selection — filter out discard_potion for next advisor call
            filtered = [a for a in actions if a != "discard_potion"]
            if filtered:
                gs = dict(gs)
                gs["available_actions"] = filtered

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
    """Load env vars (like OPENAI_API_KEY) from .mcp.json if present."""
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
        "--local", action="store_true",
        help="Use local Ollama model instead of OpenAI API",
    )
    parser.add_argument(
        "--model", type=str, default=None,
        help="Override advisor model (e.g. qwen3:8b, gpt-4o-mini)",
    )
    args = parser.parse_args()

    # Set env vars for local mode before Runner init
    if args.local:
        os.environ.setdefault("STS2_ADVISOR_BASE_URL", "http://localhost:11434/v1")
        os.environ.setdefault("STS2_ADVISOR_MODEL", "qwen3:8b")
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
