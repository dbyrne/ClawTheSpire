"use client";

import { useEffect, useState, useCallback } from "react";
import useSWR, { useSWRConfig } from "swr";
import { fetcher } from "../../lib/api";

// ---------------------------------------------------------------------------
// Types — mirror the JSON written by scripts/build_label_pool.py
// ---------------------------------------------------------------------------

type Action = {
  slot: number;
  type: string;
  card_id: string;
  card_vocab_id: number;
  mcts_visits: number;
  mcts_q: number;
  net_prob: number;
  is_network_argmax: boolean;
  is_mcts_played: boolean;
};

type Decision = {
  id: string;
  source: string;
  combat_index: number;
  step_index: number;
  state: any;
  actions: Action[];
  network_argmax_slot: number;
  mcts_played_slot: number;
  mcts_value: number;
  policy_mcts_disagree: boolean;
};

type NextResp = { decision: Decision | null; remaining: number };
type PoolListResp = { pools: Array<{ name: string; total: number; meta: any }>; total_labeled: number };

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function LabelPage() {
  const { mutate } = useSWRConfig();
  const { data: pools } = useSWR<PoolListResp>("/api/labels/pools", fetcher);
  const [selected, setSelected] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  // Auto-pick first pool when list loads
  useEffect(() => {
    if (!selected && pools?.pools?.length) {
      setSelected(pools.pools[0].name);
    }
  }, [pools, selected]);

  const nextUrl = selected ? `/api/labels/pool/${selected}/next` : null;
  const { data: nextResp, isLoading } = useSWR<NextResp>(nextUrl, fetcher, {
    revalidateOnFocus: false,
  });
  const decision = nextResp?.decision ?? null;
  const remaining = nextResp?.remaining ?? 0;

  const submitLabel = useCallback(
    async (label: "bad" | "skip", badActionSlot?: number) => {
      if (!decision || submitting) return;
      setSubmitting(true);
      try {
        const body = {
          decision_id: decision.id,
          label,
          bad_action_slot: badActionSlot ?? null,
        };
        await fetch("/api/labels/submit", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        await mutate(nextUrl);
        await mutate("/api/labels/pools");
      } finally {
        setSubmitting(false);
      }
    },
    [decision, submitting, mutate, nextUrl],
  );

  // Keyboard shortcuts: B for "bad on the network's argmax pick", S/Space for skip
  useEffect(() => {
    function onKey(e: KeyboardEvent) {
      if (!decision) return;
      if (e.key === "b" || e.key === "B") {
        e.preventDefault();
        submitLabel("bad", decision.network_argmax_slot);
      } else if (e.key === "s" || e.key === "S" || e.key === " ") {
        e.preventDefault();
        submitLabel("skip");
      }
    }
    window.addEventListener("keydown", onKey);
    return () => window.removeEventListener("keydown", onKey);
  }, [decision, submitLabel]);

  return (
    <div className="pb-24">
      <header className="flex items-center justify-between mb-3">
        <h1 className="text-xl font-bold">Label</h1>
        <div className="text-xs mono text-muted">
          {pools ? (
            <>
              <span className="text-text">{pools.total_labeled}</span> labeled ·{" "}
              <span className="text-text">{remaining}</span> remaining
            </>
          ) : (
            "loading…"
          )}
        </div>
      </header>

      {pools && pools.pools.length > 1 && (
        <div className="mb-4 flex gap-2">
          {pools.pools.map((p) => (
            <button
              key={p.name}
              onClick={() => setSelected(p.name)}
              className={`px-2 py-1 text-xs rounded border ${
                selected === p.name
                  ? "border-accent text-accent"
                  : "border-border text-muted"
              }`}
            >
              {p.name} ({p.total})
            </button>
          ))}
        </div>
      )}

      {!pools?.pools?.length && (
        <p className="text-sm text-muted">
          No label pools found. Generate one with{" "}
          <code className="mono text-text">scripts/build_label_pool.py</code>.
        </p>
      )}

      {isLoading && <p className="text-sm text-muted">loading…</p>}

      {!isLoading && !decision && pools?.pools?.length ? (
        <div className="border border-border rounded p-4 text-center text-muted">
          🎉 Pool exhausted — all decisions reviewed.
        </div>
      ) : null}

      {decision && <DecisionView decision={decision} onSubmit={submitLabel} disabled={submitting} />}

      <footer className="mt-4 text-xs text-muted">
        Shortcuts: <kbd className="border border-border px-1">B</kbd> mark
        network's pick bad ·{" "}
        <kbd className="border border-border px-1">S</kbd> skip ·{" "}
        Click an action card to mark a specific action bad.
      </footer>
    </div>
  );
}

// ---------------------------------------------------------------------------
// One decision — state + action grid
// ---------------------------------------------------------------------------

function DecisionView({
  decision,
  onSubmit,
  disabled,
}: {
  decision: Decision;
  onSubmit: (label: "bad" | "skip", slot?: number) => void;
  disabled: boolean;
}) {
  const state = decision.state ?? {};
  const player = state.player ?? {};
  const enemies = state.enemies ?? [];
  const turn = state.turn ?? "?";
  const totalVisits = decision.actions.reduce((s, a) => s + a.mcts_visits, 0);

  return (
    <div className="space-y-3">
      <div className="text-xs text-muted mono">
        {decision.id} · turn {turn} · MCTS root V= {decision.mcts_value.toFixed(2)}
      </div>

      <PlayerPanel player={player} />
      <EnemiesPanel enemies={enemies} />

      <div>
        <h2 className="text-sm font-bold mb-1">Actions ({decision.actions.length})</h2>
        <div className="space-y-1">
          {decision.actions.map((a) => (
            <ActionRow
              key={a.slot}
              action={a}
              card={lookupCard(player.hand ?? [], a.card_id)}
              totalVisits={totalVisits}
              onMarkBad={() => onSubmit("bad", a.slot)}
              disabled={disabled}
            />
          ))}
        </div>
      </div>

      <div className="flex gap-2 mt-4 sticky bottom-20 bg-panel/95 backdrop-blur py-2 -mx-2 px-2 rounded">
        <button
          disabled={disabled}
          onClick={() => onSubmit("bad", decision.network_argmax_slot)}
          className="flex-1 py-3 rounded bg-red-600/30 border border-red-500 text-red-100 font-bold disabled:opacity-50"
        >
          BAD (network's pick) [B]
        </button>
        <button
          disabled={disabled}
          onClick={() => onSubmit("skip")}
          className="flex-1 py-3 rounded bg-zinc-800 border border-border text-text font-bold disabled:opacity-50"
        >
          SKIP [S]
        </button>
      </div>
    </div>
  );
}

function PlayerPanel({ player }: { player: any }) {
  const hand = player.hand ?? [];
  return (
    <div className="border border-border rounded p-2 bg-panel/50">
      <div className="flex justify-between text-sm">
        <span className="font-bold">Player</span>
        <span className="mono">
          HP {player.hp}/{player.max_hp} · Block {player.block ?? 0} · Energy{" "}
          {player.energy ?? "?"}
        </span>
      </div>
      {Object.keys(player.powers ?? {}).length > 0 && (
        <div className="text-xs text-muted mt-1">
          Powers:{" "}
          {Object.entries(player.powers ?? {})
            .map(([k, v]) => `${k} ${v}`)
            .join(", ")}
        </div>
      )}
      <div className="mt-2 text-xs">
        <div className="text-muted mb-1">Hand ({hand.length}):</div>
        <div className="grid grid-cols-1 sm:grid-cols-2 gap-1">
          {hand.map((c: any, i: number) => (
            <div
              key={i}
              className="mono px-2 py-1 bg-zinc-800 rounded border border-border"
            >
              <div className="flex justify-between items-baseline">
                <span className="font-bold">
                  {c.name ?? c.id}
                  {c.upgraded ? "+" : ""}
                </span>
                <span className="text-muted text-[10px]">
                  {c.is_x_cost ? "X" : (c.cost ?? "?")}E ·{" "}
                  {(c.card_type ?? "?")[0]}
                </span>
              </div>
              <CardEffectsLine card={c} />
            </div>
          ))}
        </div>
      </div>
      <div className="mt-1 text-xs text-muted mono">
        Draw {(player.draw_pile ?? player.draw ?? []).length} · Discard{" "}
        {(player.discard_pile ?? player.discard ?? []).length} · Exhaust{" "}
        {(player.exhaust_pile ?? player.exhaust ?? []).length}
      </div>
    </div>
  );
}

function EnemiesPanel({ enemies }: { enemies: any[] }) {
  if (!enemies?.length) return null;
  return (
    <div className="border border-border rounded p-2 bg-panel/50">
      <div className="text-sm font-bold mb-1">Enemies</div>
      <div className="space-y-1 text-xs">
        {enemies.map((e: any, i: number) => (
          <div key={i} className="flex justify-between mono">
            <span>
              [{i}] {e.id ?? "?"}
              {(e.hp ?? 0) <= 0 ? " (dead)" : ""}
            </span>
            <span className="text-muted">
              HP {e.hp ?? "?"}/{e.max_hp ?? "?"} · Block {e.block ?? 0} ·{" "}
              {e.intent_type ?? "?"}{" "}
              {e.intent_damage ? `${e.intent_damage}×${e.intent_hits ?? 1}` : ""}
              {Object.keys(e.powers ?? {}).length > 0 && (
                <> · {Object.entries(e.powers).map(([k, v]) => `${k}${v}`).join(",")}</>
              )}
            </span>
          </div>
        ))}
      </div>
    </div>
  );
}

function ActionRow({
  action,
  card,
  totalVisits,
  onMarkBad,
  disabled,
}: {
  action: Action;
  card: any | null;
  totalVisits: number;
  onMarkBad: () => void;
  disabled: boolean;
}) {
  const visitFrac = totalVisits > 0 ? action.mcts_visits / totalVisits : 0;
  const isHighlighted = action.is_network_argmax || action.is_mcts_played;
  const verb =
    action.type === "play_card"
      ? "play"
      : action.type === "choose_card"
        ? "discard"
        : action.type === "use_potion"
          ? "use"
          : "";
  return (
    <button
      onClick={onMarkBad}
      disabled={disabled}
      className={`w-full text-left px-2 py-1.5 rounded border transition-colors disabled:opacity-50 ${
        isHighlighted
          ? "border-accent bg-accent/10"
          : "border-border bg-panel/50 hover:border-red-500/50"
      }`}
      title="Click to mark THIS specific action as bad"
    >
      <div className="flex items-center gap-2">
        <span className="mono text-xs text-muted w-6">[{action.slot}]</span>
        <span className="mono text-sm flex-1 truncate">
          {action.type === "end_turn" ? (
            <span className="italic">end turn</span>
          ) : (
            <>
              {verb} <span className="font-bold">{card?.name ?? action.card_id}</span>
              {card?.upgraded ? "+" : ""}
            </>
          )}
        </span>
        <div className="flex items-center gap-2 text-xs mono shrink-0">
          {action.is_network_argmax && (
            <span className="text-accent font-bold">⭐NET</span>
          )}
          {action.is_mcts_played && (
            <span className="text-yellow-400 font-bold">▶MCTS</span>
          )}
          <span className="w-12 text-right">
            net {(action.net_prob * 100).toFixed(0)}%
          </span>
          <span className="w-12 text-right">
            mcts {(visitFrac * 100).toFixed(0)}%
          </span>
          <span className="w-10 text-right text-muted">
            q {action.mcts_q.toFixed(2)}
          </span>
        </div>
      </div>
      {card && action.type !== "end_turn" && (
        <div className="ml-8 mt-0.5 text-[11px] text-muted">
          <CardEffectsLine card={card} />
        </div>
      )}
    </button>
  );
}

// ---------------------------------------------------------------------------
// Card effect rendering — turns the structured card JSON into a one-line
// "what does this do?" summary so the labeler doesn't need every card
// memorized. Mirrors what the in-game card text would say.
// ---------------------------------------------------------------------------

function CardEffectsLine({ card }: { card: any }) {
  if (!card) return null;
  const parts: string[] = [];

  // Damage
  if (card.damage && card.damage > 0) {
    const hits = card.hit_count ?? 1;
    const target = formatTarget(card.target);
    parts.push(
      hits > 1
        ? `${card.damage}×${hits} dmg${target ? ` ${target}` : ""}`
        : `${card.damage} dmg${target ? ` ${target}` : ""}`,
    );
  }

  // Block
  if (card.block && card.block > 0) {
    parts.push(`+${card.block} block`);
  }

  // Energy gain
  if (card.energy_gain && card.energy_gain !== 0) {
    parts.push(`${card.energy_gain > 0 ? "+" : ""}${card.energy_gain}E`);
  }

  // Card draw
  if (card.cards_draw && card.cards_draw > 0) {
    parts.push(`draw ${card.cards_draw}`);
  }

  // Self HP loss (e.g. Hemokinesis)
  if (card.hp_loss && card.hp_loss > 0) {
    parts.push(`lose ${card.hp_loss} HP`);
  }

  // Powers applied (Vulnerable, Weak, Poison, Strength, etc.)
  for (const pa of card.powers_applied ?? []) {
    if (Array.isArray(pa) && pa.length >= 2) {
      const [name, amt] = pa;
      const sign = amt > 0 ? "+" : "";
      parts.push(`${sign}${amt} ${shortPower(name)}`);
    }
  }

  // Spawned cards (e.g., Knife Trap → SHIV, Wraith Form → Smoke)
  for (const sp of card.spawns_cards ?? []) {
    parts.push(`spawn ${sp}`);
  }

  // Tags + keywords (Strike tag = synergizes with Pummel/etc.; Sly = discard sub-bonus)
  const tags = (card.tags ?? []).filter(Boolean);
  const keywords = (card.keywords ?? []).filter(Boolean);
  const annotations = [...tags, ...keywords].filter(
    (t: string) => !["Defend", "Strike"].includes(t), // hide redundant generic tags
  );

  return (
    <span>
      {parts.length > 0 ? parts.join(" · ") : <span className="italic">no direct effect</span>}
      {annotations.length > 0 && (
        <span className="ml-2 opacity-60">
          {annotations.map((a: string) => `[${a}]`).join(" ")}
        </span>
      )}
      {card.is_x_cost && <span className="ml-2 opacity-60">[X-cost: scales with energy]</span>}
    </span>
  );
}

function formatTarget(target: string | undefined): string {
  switch (target) {
    case "AnyEnemy":
      return "→ enemy";
    case "AllEnemies":
      return "→ all enemies";
    case "RandomEnemy":
      return "→ random";
    case "Self":
      return "(self)";
    default:
      return "";
  }
}

function shortPower(name: string): string {
  // Common power abbreviations to keep the line dense
  const map: Record<string, string> = {
    Vulnerable: "Vuln",
    Strength: "Str",
    Dexterity: "Dex",
    Poison: "Poison",
    Weak: "Weak",
    Frail: "Frail",
    Intangible: "Intangible",
    Artifact: "Artifact",
    Block: "Block",
    Plated_Armor: "Plate",
    Thorns: "Thorns",
  };
  return map[name] ?? name;
}

function lookupCard(hand: any[], cardId: string): any | null {
  if (!cardId || cardId === "<PAD>") return null;
  // First match by id; multiple copies in hand are functionally identical
  // for labeling purposes, so we don't bother distinguishing them.
  return hand.find((c: any) => c?.id === cardId) ?? null;
}
