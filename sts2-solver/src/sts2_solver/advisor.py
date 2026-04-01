"""Strategic advisor for non-combat decisions using OpenAI GPT-4o-mini."""

from __future__ import annotations

import json
import os
import time
import traceback
from dataclasses import dataclass
from typing import TYPE_CHECKING

from .advisor_prompts import (
    AUTO_ACTIONS,
    SYSTEM_PROMPT,
    build_user_message,
    detect_screen_type,
)
from .game_client import GameClient
from .game_data import GameDataDB

if TYPE_CHECKING:
    from .run_logger import RunLogger


DEFAULT_MODEL = os.environ.get("STS2_ADVISOR_MODEL", "gpt-4o-mini")
DEFAULT_MAX_TOKENS = int(os.environ.get("STS2_ADVISOR_MAX_TOKENS", "256"))


@dataclass
class AdvisorDecision:
    action: str
    option_index: int | None
    reasoning: str


class StrategicAdvisor:
    """Makes non-combat strategic decisions by calling OpenAI API."""

    def __init__(
        self,
        game_data: GameDataDB,
        client: GameClient,
        model: str = DEFAULT_MODEL,
        logger: RunLogger | None = None,
    ):
        self.game_data = game_data
        self.client = client
        self.model = model
        self.logger = logger
        self._openai_client = None

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            self._openai_client = OpenAI()
        return self._openai_client

    def advise(self, game_state: dict, execute: bool = True) -> str:
        """Get strategic advice for a non-combat screen.

        Args:
            game_state: Full game state dict from the game API.
            execute: If True, execute the recommended action.

        Returns:
            Formatted result string.
        """
        actions = game_state.get("available_actions", [])
        if not actions:
            return "No available actions."

        # Check if this is a combat screen
        screen = game_state.get("screen", "")
        if "COMBAT" in screen.upper() and "play_card" in actions:
            return "This is a combat screen — use solve_combat instead."

        screen_type = detect_screen_type(actions)

        if self.logger:
            self.logger.ensure_run(game_state)

        # Handle auto-actions (no LLM needed)
        if screen_type == "auto":
            return self._handle_auto(game_state, actions, execute)

        # Build prompt and call LLM
        screen_type, user_message = build_user_message(game_state, self.game_data)

        t0 = time.perf_counter()
        try:
            raw_response = self._call_llm(SYSTEM_PROMPT, user_message)
        except Exception as e:
            return f"Error calling LLM: {e}\n{traceback.format_exc()}"
        latency_ms = (time.perf_counter() - t0) * 1000

        try:
            decision = self._parse_response(raw_response)
        except Exception as e:
            return (
                f"Error parsing LLM response: {e}\n"
                f"Raw response: {raw_response}"
            )

        # Validate the action is available
        if decision.action not in actions:
            return (
                f"LLM recommended action '{decision.action}' which is not available.\n"
                f"Available: {', '.join(actions)}\n"
                f"Reasoning: {decision.reasoning}"
            )

        # Log the decision
        if self.logger:
            self.logger.log_decision(
                game_state=game_state,
                screen_type=screen_type,
                options=actions,
                choice={
                    "action": decision.action,
                    "option_index": decision.option_index,
                    "reasoning": decision.reasoning,
                },
                source="advisor",
                latency_ms=latency_ms,
            )

        return self._format_result(decision, screen_type, game_state, execute)

    def _handle_auto(self, game_state: dict, actions: list[str], execute: bool) -> str:
        """Handle screens that don't need LLM advice."""
        # Pick the first auto-action available
        for action in actions:
            if action in AUTO_ACTIONS:
                decision = AdvisorDecision(
                    action=action,
                    option_index=None,
                    reasoning="Automatic action (no decision needed)",
                )
                if self.logger:
                    self.logger.log_decision(
                        game_state=game_state,
                        screen_type="auto",
                        options=actions,
                        choice={"action": action, "option_index": None},
                        source="auto",
                    )
                return self._format_result(decision, "auto", game_state, execute)

        return "No auto-action found."

    def _call_llm(self, system: str, user: str) -> str:
        """Call OpenAI API and return the response text."""
        client = self._get_openai_client()
        response = client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
            max_tokens=DEFAULT_MAX_TOKENS,
            temperature=0.3,
            response_format={"type": "json_object"},
        )
        return response.choices[0].message.content

    def _parse_response(self, raw: str) -> AdvisorDecision:
        """Parse the JSON response from the LLM."""
        # Strip code fences if present
        text = raw.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        data = json.loads(text)
        return AdvisorDecision(
            action=data["action"],
            option_index=data.get("option_index"),
            reasoning=data.get("reasoning", ""),
        )

    def _format_result(
        self,
        decision: AdvisorDecision,
        screen_type: str,
        game_state: dict,
        execute: bool,
    ) -> str:
        """Format the result and optionally execute."""
        lines = [
            f"Screen: {screen_type.upper()}",
            f"Decision: {decision.action}"
            + (f" (option_index={decision.option_index})" if decision.option_index is not None else ""),
            f"Reasoning: {decision.reasoning}",
        ]

        if not execute:
            lines.append("")
            lines.append("(dry run — not executed)")
            return "\n".join(lines)

        # Execute the action
        lines.append("")
        lines.append("Executing:")
        try:
            self.client.execute_action(
                decision.action,
                option_index=decision.option_index,
            )
            lines.append(f"  {decision.action} -> OK")
        except Exception as e:
            lines.append(f"  {decision.action} -> FAILED: {e}")

        return "\n".join(lines)
