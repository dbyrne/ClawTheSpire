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
from .game_data import strip_markup
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
        self._card_reward_handled = False  # Reset when leaving reward screen
        self._deck_select_stuck = False  # Track stuck deck_select screens
        self._stuck_since: float | None = None  # Timestamp when we got stuck
        self._shop_visited = False  # Prevent re-opening shop after closing
        self._last_floor: int | None = None  # Track floor for shop reset
        self._last_screen_key: tuple[str, str] | None = None  # (screen, screen_type)
        self._screen_repeat_count: int = 0  # Same-screen repeat counter

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

        # Reward screen: collect_rewards_and_proceed auto-picks the first
        # card reward — NEVER use it when an unhandled card choice exists.
        # Instead, claim the card reward item to open the selection screen,
        # then let the advisor choose or skip.
        if "collect_rewards_and_proceed" in actions and screen_type != "card_reward":
            reward = gs.get("reward") or {}
            if not reward:
                reward = (gs.get("agent_view") or {}).get("reward") or {}

            has_card_choice = (
                reward.get("pending_card_choice")
                or "choose_reward_card" in actions
                or "skip_reward_cards" in actions
            )

            # Check reward items for card-type rewards.
            # Raw state: reward_type="Card"; agent_view: line="card: ...".
            reward_items = reward.get("rewards") or []
            has_card_reward_item = any(
                self._is_card_reward_item(item)
                for item in reward_items
                if item.get("claimable", True)
            )

            # If reward data is empty but we just arrived at the reward screen,
            # wait for the data to populate before auto-proceeding.
            if not reward_items and not has_card_choice and "claim_reward" in actions:
                return  # Let next tick re-check once reward data is populated

            # If we already handled the card choice this reward screen,
            # don't try to open it again — just proceed.
            if self._card_reward_handled:
                has_card_reward_item = False
                has_card_choice = False

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
                # No card choice (or already handled) — safe to auto-proceed
                self._card_reward_handled = False  # Reset for next reward
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
            # Decision keywords that need advisor input
            is_decision = any(kw in prompt for kw in (
                "choose", "remove", "upgrade", "transform", "add", "select",
            ))
            if is_decision:
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

        # LLM-based decision
        try:
            result_str = self.advisor.advise(gs, execute=not self.dry_run)
        except Exception as e:
            self._log_action(f"[red]Advisor error: {e}[/red]")
            return

        # If card_reward advisor mentions empty/no options, skip and retry next tick
        if screen_type == "card_reward" and any(
            phrase in result_str.lower()
            for phrase in ("no card", "empty", "0 cards", "no options")
        ):
            self._log_action("  [dim]Card reward empty — retrying next tick[/dim]")
            return

        # If the advisor recommended an invalid/failed action, fall back to a safe default
        if "not available" in result_str or "FAILED" in result_str:
            self._log_action(f"  [yellow]Invalid action — falling back[/yellow]")
            # Ordered list of fallback actions to try
            _FALLBACKS = [
                ("choose_map_node", 0),
                ("choose_event_option", 0),
                ("choose_rest_option", 0),
                ("close_shop_inventory", None),
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
                if fb_action == "close_shop_inventory":
                    self._shop_visited = True
                self._log_action(f"  [dim]auto: {fb_action}({fb_idx}) (fallback)[/dim]")
                self.action_count += 1
                return
            return  # Nothing we can do, let next tick try

        # Mark card reward as handled so we don't re-open it
        if screen_type == "card_reward":
            self._card_reward_handled = True

        # Track shop visits to prevent re-opening after closing
        if screen_type == "shop" and "close_shop_inventory" in result_str and "-> OK" in result_str:
            self._shop_visited = True

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

        For single-select (upgrade, transform): use the advisor.
        For multi-select (e.g. "Choose 2 to Remove"): pick deterministically
        using Strikes first, then Defends, never Bash — since the advisor
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
        """Single-select deck screen — use advisor."""
        # Filter discard_potion from actions
        filtered_actions = [a for a in gs.get("available_actions", []) if a != "discard_potion"]
        if filtered_actions:
            gs = dict(gs)
            gs["available_actions"] = filtered_actions

        try:
            result_str = self.advisor.advise(gs, execute=not self.dry_run)
        except Exception as e:
            self._log_action(f"[red]Advisor error: {e}[/red]")
            return

        lines = result_str.split("\n")
        decision_line = next(
            (l for l in lines if l.startswith("Decision:")),
            lines[0] if lines else "?",
        )
        self._log_action(f"  [blue]{decision_line}[/blue]")
        self.action_count += 1
        self._refresh()

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

        For remove: Strikes first, then Defends, never Bash.
        For other multi-selects: pick sequentially from index 0.
        """
        import re
        multi_match = re.search(r"choose\s+(\d+)", prompt_text)
        num_to_pick = int(multi_match.group(1)) if multi_match else 2

        is_remove = "remove" in prompt_text
        is_upgrade = "upgrade" in prompt_text

        # Build priority order for indices
        if is_remove:
            # Remove Strikes first, then Defends, then others, NEVER Bash
            priority = []
            for card in cards:
                name = (card.get("name") or "").lower()
                idx = card.get("index", 0)
                if "bash" in name:
                    continue  # Never remove Bash
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
