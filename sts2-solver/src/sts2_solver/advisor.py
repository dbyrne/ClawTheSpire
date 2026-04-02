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
DEFAULT_BASE_URL = os.environ.get("STS2_ADVISOR_BASE_URL", "")  # empty = OpenAI default


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
        base_url: str = DEFAULT_BASE_URL,
        logger: RunLogger | None = None,
    ):
        self.game_data = game_data
        self.client = client
        self.model = model
        self.base_url = base_url
        self.logger = logger
        self._openai_client = None

    @property
    def is_local(self) -> bool:
        """True if using a local model (Ollama, vLLM, etc.)."""
        return bool(self.base_url)

    def _get_openai_client(self):
        if self._openai_client is None:
            from openai import OpenAI
            kwargs = {}
            if self.base_url:
                kwargs["base_url"] = self.base_url
                kwargs["api_key"] = "ollama"  # Ollama doesn't need a real key
            self._openai_client = OpenAI(**kwargs)
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

        # Coerce null option_index to 0 for actions that require an index.
        # Special case: choose_reward_card with null means "skip" — convert
        # to skip_reward_cards if available instead of accidentally taking a card.
        if decision.option_index is None and decision.action == "choose_reward_card":
            if "skip_reward_cards" in actions:
                decision = AdvisorDecision(
                    action="skip_reward_cards", option_index=None,
                    reasoning=decision.reasoning,
                )
            else:
                decision = AdvisorDecision(
                    action=decision.action, option_index=0,
                    reasoning=decision.reasoning,
                )
        elif decision.option_index is None and decision.action in {
            "choose_map_node", "choose_event_option", "choose_rest_option",
            "choose_treasure_relic", "select_deck_card",
            "buy_card", "buy_relic", "buy_potion", "claim_reward",
        }:
            decision = AdvisorDecision(
                action=decision.action, option_index=0, reasoning=decision.reasoning,
            )

        # Validate the action is available — try to fix common LLM mistakes
        if decision.action not in actions:
            # Fuzzy-match: LLM often returns a shortened or wrong action name
            _ACTION_ALIASES = {
                "proceed": "collect_rewards_and_proceed",
                "skip": "skip_reward_cards",
                "end_turn": "end_turn",
                "close_shop": "close_shop_inventory",
                "leave": "close_shop_inventory",
            }
            fixed = _ACTION_ALIASES.get(decision.action)
            if fixed and fixed in actions:
                decision = AdvisorDecision(
                    action=fixed, option_index=decision.option_index,
                    reasoning=decision.reasoning,
                )
            else:
                # Log the failure for debugging
                if self.logger:
                    self.logger.log_decision(
                        game_state=game_state,
                        screen_type=screen_type,
                        options=actions,
                        choice={
                            "action": decision.action,
                            "option_index": decision.option_index,
                            "reasoning": decision.reasoning,
                            "error": "action_not_available",
                            "raw_response": raw_response[:500],
                        },
                        source="advisor_error",
                        latency_ms=latency_ms,
                        user_prompt=user_message,
                    )
                return (
                    f"LLM recommended action '{decision.action}' which is not available.\n"
                    f"Available: {', '.join(actions)}\n"
                    f"Reasoning: {decision.reasoning}"
                )

        # Log the decision (include user_prompt for training data)
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
                user_prompt=user_message,
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
        """Call LLM API and return the response text."""
        client = self._get_openai_client()

        # Qwen3 uses thinking mode by default — disable it for faster,
        # direct JSON responses by prepending /no_think
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        if self.is_local and "qwen3" in self.model.lower():
            messages[-1] = {
                "role": "user",
                "content": "/no_think\n" + user,
            }

        kwargs: dict = {
            "model": self.model,
            "messages": messages,
            "max_tokens": DEFAULT_MAX_TOKENS,
            "temperature": 0.3,
        }
        # JSON mode: OpenAI and some Ollama models support it
        try:
            kwargs["response_format"] = {"type": "json_object"}
            response = client.chat.completions.create(**kwargs)
        except Exception:
            # Fall back without JSON mode (some local models don't support it)
            kwargs.pop("response_format", None)
            response = client.chat.completions.create(**kwargs)

        content = response.choices[0].message.content or ""

        # Ollama OpenAI-compat may put Qwen3 thinking in 'reasoning' field
        # with empty content — fall back to native API
        if not content.strip() and self.is_local:
            content = self._call_ollama_native(messages)

        return content

    def _call_ollama_native(self, messages: list[dict]) -> str:
        """Fall back to Ollama native API when OpenAI compat fails."""
        import urllib.request
        import urllib.error

        base = self.base_url.replace("/v1", "")
        url = f"{base}/api/chat"
        payload = json.dumps({
            "model": self.model,
            "messages": messages,
            "stream": False,
            "options": {"temperature": 0.3, "num_predict": DEFAULT_MAX_TOKENS},
        }).encode()
        req = urllib.request.Request(url, data=payload, method="POST")
        req.add_header("Content-Type", "application/json")
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
        return data.get("message", {}).get("content", "")

    def _parse_response(self, raw: str) -> AdvisorDecision:
        """Parse the JSON response from the LLM.

        Handles: bare JSON, code-fenced JSON, JSON embedded in text,
        and Qwen3's <think>...</think> blocks.
        """
        text = raw.strip()

        # Strip Qwen3 thinking blocks
        if "<think>" in text:
            import re
            text = re.sub(r"<think>.*?</think>", "", text, flags=re.DOTALL).strip()

        # Strip code fences
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0].strip()

        # Try direct parse first
        try:
            data = json.loads(text)
        except json.JSONDecodeError:
            # Extract first JSON object from freeform text
            start = text.find("{")
            end = text.rfind("}") + 1
            if start >= 0 and end > start:
                data = json.loads(text[start:end])
            else:
                raise

        # Coerce option_index to int (LLM sometimes returns "2" instead of 2)
        raw_idx = data.get("option_index")
        if raw_idx is not None:
            try:
                raw_idx = int(raw_idx)
            except (ValueError, TypeError):
                raw_idx = None

        return AdvisorDecision(
            action=data["action"],
            option_index=raw_idx,
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
