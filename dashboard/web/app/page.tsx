"use client";

import { useMemo, useState, useEffect, useCallback, useRef } from "react";
import Link from "next/link";
import { useLocalSocket } from "@/lib/useLocalSocket";
import type { Run, RunEvent } from "@/lib/types";
import {
  AreaChart,
  Area,
  ComposedChart,
  Bar,
  Cell,
  Line,
  LineChart,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
  Legend,
} from "recharts";

// ---------------------------------------------------------------------------
// Training Progress Types
// ---------------------------------------------------------------------------

interface TrainingProgress {
  generation: number;
  num_generations: number;
  games_played: number;
  win_rate: number;
  gen_win_rate: number;
  gen_avg_floor: number;
  gen_min_floor: number;
  gen_max_floor: number;
  value_head_mean: number;
  value_head_min: number;
  value_head_max: number;
  value_head_spread: number;
  buffer_size: number;
  total_loss: number;
  value_loss: number;
  policy_loss: number;
  option_loss: number;
  option_buffer_size: number;
  lr: number;
  mcts_sims: number;
  games_per_gen: number;
  elapsed: string;
  gen_time: number;
  status: string;
  timestamp: number;
}

interface TrainingHistoryPoint {
  gen: number;
  total_loss: number;
  value_loss: number;
  policy_loss: number;
  option_loss: number;
  win_rate: number;
  gen_win_rate: number;
  gen_avg_floor: number;
}

// ---------------------------------------------------------------------------
// Training Progress Hook
// ---------------------------------------------------------------------------

function useTrainingProgress(pollMs = 30_000) {
  const [progress, setProgress] = useState<TrainingProgress | null>(null);
  const [history, setHistory] = useState<TrainingHistoryPoint[]>([]);
  const seenGens = useRef(new Set<number>());
  const historyLoaded = useRef(false);

  // Load full history from JSONL file on mount
  useEffect(() => {
    (async () => {
      try {
        const res = await fetch("/api/training?type=history");
        if (!res.ok) return;
        const data: TrainingHistoryPoint[] = await res.json();
        if (data.length > 0) {
          setHistory(data);
          for (const pt of data) seenGens.current.add(pt.gen);
        }
      } catch {
        /* ignore */
      }
      historyLoaded.current = true;
    })();
  }, []);

  const fetchProgress = useCallback(async () => {
    try {
      const res = await fetch("/api/training");
      if (!res.ok) return;
      const data: TrainingProgress = await res.json();
      if (!data) return;
      setProgress(data);

      // Append new point if we haven't seen this gen yet
      if (historyLoaded.current && !seenGens.current.has(data.generation)) {
        seenGens.current.add(data.generation);
        setHistory((prev) => [
          ...prev,
          {
            gen: data.generation,
            total_loss: data.total_loss,
            value_loss: data.value_loss,
            policy_loss: data.policy_loss,
            option_loss: data.option_loss,
            win_rate: data.win_rate,
            gen_win_rate: data.gen_win_rate,
            gen_avg_floor: data.gen_avg_floor,
          },
        ]);
      }
    } catch {
      /* ignore */
    }
  }, []);

  useEffect(() => {
    fetchProgress();
    const id = setInterval(fetchProgress, pollMs);
    return () => clearInterval(id);
  }, [fetchProgress, pollMs]);

  return { progress, history };
}

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------

const GEN_PALETTE = [
  "#ffaa00", "#ff2255", "#00ddff", "#00ff88", "#bb66ff",
  "#ff6622", "#ff44aa", "#00ccdd", "#ffcc44", "#aa7722",
  "#00ddaa", "#dd66ff", "#ff3355", "#88ee00", "#6655ff",
];

function genColor(gen: string | null): string {
  if (!gen) return "#ffaa00";
  const m = gen.match(/(\d+)/);
  if (!m) return "#ffaa00";
  const idx = (parseInt(m[1], 10) - 1) % GEN_PALETTE.length;
  return GEN_PALETTE[idx < 0 ? 0 : idx];
}

const ACT_BOSSES = [
  { floor: 17, label: "Act 2" },
  { floor: 34, label: "Act 3" },
  { floor: 52, label: "Final Boss" },
];

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

function movingAvg(data: number[], window: number): (number | null)[] {
  return data.map((_, i) => {
    if (i < window - 1) return null;
    const slice = data.slice(i - window + 1, i + 1);
    return slice.reduce((a, b) => a + b, 0) / slice.length;
  });
}

/** Map network value (-1..+1) to win probability (0..100) */
function toWinPct(nv: number): number {
  return ((nv + 1) / 2) * 100;
}

/** Color for a win probability value — neon scheme */
function winPctColor(pct: number): string {
  if (pct >= 65) return "#00ff88";
  if (pct >= 50) return "#00ddff";
  if (pct >= 35) return "#ffaa00";
  if (pct >= 20) return "#ff6622";
  return "#ff2255";
}

// ---------------------------------------------------------------------------
// Win Expectancy Gauge — neon arc
// ---------------------------------------------------------------------------

function WinGauge({ value }: { value: number | null }) {
  const pct = value != null ? toWinPct(value) : null;
  const fraction = pct != null ? pct / 100 : 0;

  const r = 72;
  const cx = 100;
  const cy = 90;
  const circumference = Math.PI * r;
  const dashOffset = circumference * (1 - fraction);
  const color = pct != null ? winPctColor(pct) : "#353c6a";

  return (
    <svg viewBox="0 0 200 108" className="w-full gauge-glow" style={{ maxWidth: 240 }}>
      <defs>
        <linearGradient id="gaugeTrack" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#ff2255" stopOpacity="0.15" />
          <stop offset="50%" stopColor="#ffaa00" stopOpacity="0.08" />
          <stop offset="100%" stopColor="#00ff88" stopOpacity="0.15" />
        </linearGradient>
        <linearGradient id="gaugeValue" x1="0" y1="0" x2="1" y2="0">
          <stop offset="0%" stopColor="#ff2255" />
          <stop offset="30%" stopColor="#ff6622" />
          <stop offset="50%" stopColor="#ffaa00" />
          <stop offset="70%" stopColor="#00ddff" />
          <stop offset="100%" stopColor="#00ff88" />
        </linearGradient>
        <filter id="gaugeBlur">
          <feGaussianBlur stdDeviation="4" />
        </filter>
      </defs>

      {/* Track */}
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        stroke="url(#gaugeTrack)"
        strokeWidth="10"
        fill="none"
        strokeLinecap="round"
      />

      {/* Glow */}
      {pct != null && (
        <path
          d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
          stroke={color}
          strokeWidth="16"
          fill="none"
          strokeLinecap="round"
          strokeDasharray={circumference}
          strokeDashoffset={dashOffset}
          className="gauge-arc"
          filter="url(#gaugeBlur)"
          opacity="0.5"
        />
      )}

      {/* Value arc */}
      <path
        d={`M ${cx - r} ${cy} A ${r} ${r} 0 0 1 ${cx + r} ${cy}`}
        stroke="url(#gaugeValue)"
        strokeWidth="8"
        fill="none"
        strokeLinecap="round"
        strokeDasharray={circumference}
        strokeDashoffset={dashOffset}
        className="gauge-arc"
      />

      {/* Tick marks */}
      {[0, 0.25, 0.5, 0.75, 1].map((t) => {
        const angle = Math.PI - t * Math.PI;
        const innerR = r - 14;
        const outerR = r - 10;
        return (
          <line
            key={t}
            x1={cx + innerR * Math.cos(angle)}
            y1={cy - innerR * Math.sin(angle)}
            x2={cx + outerR * Math.cos(angle)}
            y2={cy - outerR * Math.sin(angle)}
            stroke="#353c6a"
            strokeWidth="1.5"
          />
        );
      })}

      {/* Center value */}
      <text x={cx} y={cy - 16} textAnchor="middle" fill={color} fontSize="48" fontWeight="800" fontFamily="var(--font-display)">
        {pct != null ? `${pct.toFixed(0)}` : "—"}
      </text>
      <text x={cx} y={cy + 2} textAnchor="middle" fill={color} fontSize="14" fontWeight="700" fontFamily="var(--font-display)" opacity="0.7">
        WIN %
      </text>
    </svg>
  );
}

// ---------------------------------------------------------------------------
// Live Run Sidebar
// ---------------------------------------------------------------------------

interface LivePoint {
  index: number;
  floor: number;
  hp: number | null;
  networkValue: number | null;
  label: string;
  eventType: string;
}

function LiveSidebar({ run, events, connected }: { run: Run; events: RunEvent[]; connected: boolean }) {
  const isLive = !run.outcome;

  const points = useMemo(() => {
    const pts: LivePoint[] = [];
    let idx = 0;
    for (const ev of events) {
      const detail = ev.detail as Record<string, unknown> | null;
      let label = ev.event_type;
      if (ev.event_type === "combat_start")
        label = `Combat: ${(detail?.enemies as string[])?.join(", ") ?? "?"}`;
      else if (ev.event_type === "combat_turn")
        label = `T${detail?.turn}: ${(detail?.cards_played as string[])?.join(", ") ?? ""}`;
      else if (ev.event_type === "combat_end")
        label = `${detail?.outcome} (${detail?.turns}t)`;
      else if (ev.event_type === "decision")
        label = `${detail?.screen_type}: ${detail?.choice ?? ""}`;
      pts.push({
        index: idx++,
        floor: ev.floor ?? 0,
        hp: ev.hp,
        networkValue: ev.network_value != null ? Math.round(ev.network_value * 100) / 100 : null,
        label,
        eventType: ev.event_type,
      });
    }
    return pts;
  }, [events]);

  const lastValue = [...points].reverse().find((p) => p.networkValue != null)?.networkValue ?? null;
  const lastEvent = points[points.length - 1];
  const currentFloor = lastEvent?.floor ?? run.final_floor ?? 0;
  const currentHp = lastEvent?.hp ?? run.final_hp ?? 0;
  const maxHp = run.max_hp ?? 80;
  const hpPct = maxHp > 0 ? (currentHp / maxHp) * 100 : 0;

  return (
    <div className="flex flex-col gap-2">
      {/* Header: character + status */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2">
          {isLive ? <div className="live-dot" /> : (
            <span className={`w-3 h-3 rounded-full ${connected ? "bg-verdant" : "bg-blood animate-pulse"}`} />
          )}
          <span className={`font-serif text-2xl font-bold tracking-wide ${
            run.character === "The Ironclad" ? "text-blood-bright text-glow-pink" : "text-verdant-bright text-glow-green"
          }`}>
            {run.character.replace("The ", "")}
          </span>
        </div>
        {run.outcome && (
          <span className={`font-display text-xs font-bold uppercase tracking-widest px-2.5 py-1 rounded ${
            run.outcome === "victory"
              ? "text-verdant-bright bg-verdant-bright/15 border border-verdant-bright/30"
              : "text-blood-bright bg-blood-bright/15 border border-blood-bright/30"
          }`}>
            {run.outcome}
          </span>
        )}
      </div>

      {/* Checkpoint & gen */}
      <div className="text-bone-faint text-xs font-mono truncate">
        {run.checkpoint ?? "—"} {run.gen ? `· ${run.gen}` : ""}
      </div>

      {/* Win Expectancy Gauge */}
      <div className="flex justify-center -my-1">
        <WinGauge value={lastValue} />
      </div>

      {/* Floor + HP side by side */}
      <div className="flex gap-2">
        {/* Floor */}
        <div className="flex-1 bg-surface/50 rounded px-3 py-2 border border-rune/80">
          <div className="text-bone-faint text-[10px] uppercase tracking-widest font-display font-bold mb-0.5">Floor</div>
          <div className="text-4xl font-black font-display text-ember leading-none text-glow-amber">{currentFloor}</div>
        </div>
        {/* HP */}
        <div className="flex-1 bg-surface/50 rounded px-3 py-2 border border-rune/80">
          <div className="text-bone-faint text-[10px] uppercase tracking-widest font-display font-bold mb-0.5">HP</div>
          <div className="text-2xl font-bold font-mono text-blood-bright leading-none">
            {currentHp}<span className="text-bone-faint text-base">/{maxHp}</span>
          </div>
          <div className="hp-bar-track mt-1.5">
            <div className="hp-bar-fill" style={{ width: `${hpPct}%` }} />
          </div>
        </div>
      </div>

      {/* Last Decision Scores */}
      {(() => {
        const lastDecision = [...events].reverse().find(
          (e) => e.event_type === "decision" && (e.detail as Record<string, unknown>)?.head_scores
        );
        if (!lastDecision) return null;
        const detail = lastDecision.detail as Record<string, unknown>;
        const hs = detail.head_scores as { head: string; chosen: number; options: { label: string; score: number }[] };
        if (!hs?.options?.length) return null;
        const maxScore = Math.max(...hs.options.map((o) => Math.abs(o.score)), 0.01);
        return (
          <>
            <div className="rune-divider" />
            <div>
              <div className="flex items-center justify-between mb-1.5">
                <span className="text-arcane text-[10px] uppercase tracking-widest font-display font-bold">
                  {hs.head === "deck_eval" ? "Deck Eval" : "Option Eval"}
                </span>
                <span className="text-bone-faint text-[11px] font-mono">
                  {(detail.screen_type as string) ?? ""}
                </span>
              </div>
              <div className="space-y-0.5">
                {hs.options
                  .map((opt, i) => ({ ...opt, i }))
                  .sort((a, b) => b.score - a.score)
                  .slice(0, 4)
                  .map((opt) => {
                    const isChosen = opt.i === hs.chosen;
                    const barW = Math.max(4, (Math.abs(opt.score) / maxScore) * 100);
                    return (
                      <div key={opt.i} className={`flex items-center gap-2 ${isChosen ? "text-bone" : "text-bone-dim"}`}>
                        <span className="w-[85px] truncate font-mono text-[12px]" title={opt.label}>
                          {isChosen ? "▸ " : "  "}{opt.label}
                        </span>
                        <div className="flex-1 h-[8px] bg-rune/40 rounded-sm overflow-hidden">
                          <div
                            className="h-full rounded-sm"
                            style={{
                              width: `${barW}%`,
                              background: isChosen
                                ? "linear-gradient(90deg, #00aadd, #00ddff)"
                                : "#353c6a",
                              boxShadow: isChosen ? "0 0 8px rgba(0,221,255,0.3)" : "none",
                            }}
                          />
                        </div>
                        <span className="w-[38px] text-right font-mono text-[12px]">{opt.score.toFixed(2)}</span>
                      </div>
                    );
                  })}
              </div>
            </div>
          </>
        );
      })()}
    </div>
  );
}

// ---------------------------------------------------------------------------
// Win Expectancy Timeline Chart
// ---------------------------------------------------------------------------

function TimelineChart({ events }: { events: RunEvent[] }) {
  const points = useMemo(() => {
    const pts: LivePoint[] = [];
    let idx = 0;
    for (const ev of events) {
      const detail = ev.detail as Record<string, unknown> | null;
      let label = ev.event_type;
      if (ev.event_type === "combat_start")
        label = `Combat: ${(detail?.enemies as string[])?.join(", ") ?? "?"}`;
      else if (ev.event_type === "combat_turn")
        label = `T${detail?.turn}: ${(detail?.cards_played as string[])?.join(", ") ?? ""}`;
      else if (ev.event_type === "combat_end")
        label = `${detail?.outcome} (${detail?.turns}t)`;
      else if (ev.event_type === "decision")
        label = `${detail?.screen_type}: ${detail?.choice ?? ""}`;
      pts.push({
        index: idx++,
        floor: ev.floor ?? 0,
        hp: ev.hp,
        networkValue: ev.network_value != null ? Math.max(0, Math.round(ev.network_value * 100) / 100) : null,
        label,
        eventType: ev.event_type,
      });
    }
    return pts;
  }, [events]);

  const combatRegions = useMemo(() => {
    const regions: { start: number; end: number }[] = [];
    let start: number | null = null;
    for (const pt of points) {
      if (pt.eventType === "combat_start") start = pt.index;
      else if (pt.eventType === "combat_end" && start != null) {
        regions.push({ start, end: pt.index });
        start = null;
      }
    }
    if (start != null) regions.push({ start, end: points.length - 1 });
    return regions;
  }, [points]);

  if (points.length === 0) {
    return (
      <div className="text-bone-faint text-center py-16 font-display text-lg tracking-widest uppercase">
        Awaiting signal from the Spire...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <AreaChart data={points} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <defs>
          <linearGradient id="valueGrad" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="#00ddff" stopOpacity={0.4} />
            <stop offset="60%" stopColor="#00ddff" stopOpacity={0.08} />
            <stop offset="100%" stopColor="#00ddff" stopOpacity={0.01} />
          </linearGradient>
        </defs>

        <CartesianGrid strokeDasharray="3 3" stroke="rgba(37,42,74,0.5)" />
        <XAxis
          dataKey="index"
          hide
        />
        <YAxis
          yAxisId="value"
          domain={[0, 1]}
          tick={{ fill: "#666688", fontSize: 12, fontFamily: "var(--font-mono)" }}
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
          axisLine={false}
          tickLine={false}
          width={48}
        />
        <YAxis
          yAxisId="hp"
          orientation="right"
          domain={[0, "dataMax"]}
          tick={{ fill: "#666688", fontSize: 12, fontFamily: "var(--font-mono)" }}
          axisLine={false}
          tickLine={false}
          width={36}
        />

        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const pt = payload[0].payload as LivePoint;
            return (
              <div className="arcane-panel px-4 py-3 text-sm" style={{ background: "#101325f5", border: "1.5px solid #353c6a" }}>
                <div className="text-bone-dim mb-1 font-display text-xs tracking-wider uppercase">Floor {pt.floor}</div>
                <div className="text-bone mb-1.5 font-mono text-sm">{pt.label}</div>
                {pt.networkValue != null && (
                  <div className="text-arcane-bright font-bold text-lg font-display text-glow-cyan">{toWinPct(pt.networkValue).toFixed(0)}% win</div>
                )}
                {pt.hp != null && <div className="text-blood-bright font-mono">HP {pt.hp}</div>}
              </div>
            );
          }}
        />

        {combatRegions.map((r, i) => (
          <ReferenceArea key={i} yAxisId="value" x1={r.start} x2={r.end} fill="rgba(255,34,85,0.1)" strokeOpacity={0} />
        ))}

        <Area
          yAxisId="value"
          type="monotone"
          dataKey="networkValue"
          stroke="#00ddff"
          strokeWidth={2.5}
          fill="url(#valueGrad)"
          connectNulls
          isAnimationActive={false}
          dot={false}
        />
        <Line
          yAxisId="hp"
          type="stepAfter"
          dataKey="hp"
          stroke="#ff5588"
          strokeWidth={2}
          strokeDasharray="6 3"
          dot={false}
          connectNulls
          isAnimationActive={false}
        />
      </AreaChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Floor Progression Chart
// ---------------------------------------------------------------------------

interface FloorPoint {
  index: number;
  runId: string;
  floor: number;
  gen: string | null;
  character: string;
  fill: string;
  avg: number | null;
}

function FloorChart({ data, bestFloor }: { data: FloorPoint[]; bestFloor: number }) {
  if (data.length === 0) {
    return <div className="text-bone-faint text-center py-8 text-sm font-display tracking-widest uppercase">No completed runs</div>;
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <ComposedChart data={data} margin={{ top: 4, right: 8, bottom: 4, left: 90 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(37,42,74,0.5)" />
        <XAxis dataKey="index" hide />
        <YAxis domain={[0, Math.max(bestFloor + 5, 55)]} hide />

        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const pt = payload[0].payload as FloorPoint;
            return (
              <div className="arcane-panel px-4 py-3 text-sm" style={{ background: "#101325f5", border: "1.5px solid #353c6a" }}>
                <div className="text-ember-bright font-bold font-display text-lg text-glow-amber">Floor {pt.floor}</div>
                <div className="text-bone-dim mt-1 font-mono">
                  {pt.character.replace("The ", "")} &middot; {pt.gen ?? "?"}
                </div>
              </div>
            );
          }}
        />

        {ACT_BOSSES.map((act) => (
          <ReferenceLine
            key={act.floor}
            y={act.floor}
            stroke="rgba(255,170,0,0.3)"
            strokeDasharray="6 4"
            label={{ value: act.label, position: "left", fill: "#ffaa00", fontSize: 13, fontFamily: "var(--font-display)" }}
          />
        ))}

        <Bar dataKey="floor" radius={[2, 2, 0, 0]} isAnimationActive={false} maxBarSize={20}>
          {data.map((entry, i) => (
            <Cell key={i} fill={entry.fill + "80"} stroke={entry.fill} strokeWidth={0.5} />
          ))}
        </Bar>
      </ComposedChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Training Stats Sidebar
// ---------------------------------------------------------------------------

function TrainingStats({ progress }: { progress: TrainingProgress | null }) {
  if (!progress) return null;

  const genPct = ((progress.generation / progress.num_generations) * 100).toFixed(1);
  const winPct = (progress.win_rate * 100).toFixed(1);
  const genWinPct = (progress.gen_win_rate * 100).toFixed(0);

  // Estimate remaining time
  const remainingGens = progress.num_generations - progress.generation;
  const etaSeconds = remainingGens * progress.gen_time;
  const etaHours = (etaSeconds / 3600).toFixed(1);

  return (
    <div className="arcane-panel p-4 animate-in flex-1 flex flex-col">
      <h3 className="font-display text-xs tracking-[0.2em] uppercase text-mystic-bright font-bold mb-3 text-glow-cyan">
        Training Progress
      </h3>

      {/* Gen progress bar */}
      <div className="mb-4">
        <div className="flex justify-between text-sm font-mono mb-1">
          <span className="text-bone">Gen {progress.generation}<span className="text-bone-dim">/{progress.num_generations}</span></span>
          <span className="text-mystic font-bold">{genPct}%</span>
        </div>
        <div className="h-2 bg-rune/60 rounded-full overflow-hidden">
          <div
            className="h-full rounded-full"
            style={{
              width: `${genPct}%`,
              background: "linear-gradient(90deg, #bb66ff, #00ddff)",
              boxShadow: "0 0 8px rgba(187,102,255,0.4)",
            }}
          />
        </div>
      </div>

      {/* Key performance stats — larger */}
      <div className="grid grid-cols-2 gap-3 mb-4">
        <div className="bg-surface/50 rounded px-3 py-2 border border-rune/80 text-center">
          <div className="text-2xl font-black font-display text-bone">{progress.games_played.toLocaleString()}</div>
          <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mt-0.5">Games</div>
        </div>
        <div className="bg-surface/50 rounded px-3 py-2 border border-rune/80 text-center">
          <div className="text-2xl font-black font-display text-verdant-bright text-glow-green">{winPct}%</div>
          <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mt-0.5">Win Rate</div>
        </div>
        <div className="bg-surface/50 rounded px-3 py-2 border border-rune/80 text-center">
          <div className="text-2xl font-black font-display text-verdant">{genWinPct}%</div>
          <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mt-0.5">Gen Win%</div>
        </div>
        <div className="bg-surface/50 rounded px-3 py-2 border border-rune/80 text-center">
          <div className="text-2xl font-black font-display text-ember text-glow-amber">{progress.gen_avg_floor.toFixed(1)}</div>
          <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mt-0.5">Avg Floor</div>
        </div>
      </div>

      <div className="rune-divider my-1" />

      {/* Losses */}
      <div className="my-3">
        <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mb-2">Network Loss</div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm font-mono">
          <div className="flex justify-between">
            <span className="text-bone-dim">Total</span>
            <span className="text-arcane-bright font-bold">{progress.total_loss.toFixed(4)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-bone-dim">Policy</span>
            <span className="text-arcane">{progress.policy_loss.toFixed(4)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-bone-dim">Value</span>
            <span className="text-arcane">{progress.value_loss.toFixed(4)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-bone-dim">Option</span>
            <span className="text-arcane">{progress.option_loss.toFixed(4)}</span>
          </div>
        </div>
      </div>

      <div className="rune-divider my-1" />

      {/* Timing */}
      <div className="my-3">
        <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mb-2">Timing</div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm font-mono">
          <div className="flex justify-between">
            <span className="text-bone-dim">Elapsed</span>
            <span className="text-bone">{progress.elapsed}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-bone-dim">Gen Time</span>
            <span className="text-bone">{progress.gen_time.toFixed(0)}s</span>
          </div>
          <div className="flex justify-between">
            <span className="text-bone-dim">ETA</span>
            <span className="text-bone">{etaHours}h</span>
          </div>
        </div>
      </div>

      <div className="rune-divider my-1" />

      {/* Config */}
      <div className="mt-3">
        <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mb-2">Config</div>
        <div className="grid grid-cols-2 gap-x-4 gap-y-2 text-sm font-mono">
          <div className="flex justify-between">
            <span className="text-bone-dim">MCTS</span>
            <span className="text-bone">{progress.mcts_sims}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-bone-dim">LR</span>
            <span className="text-bone">{progress.lr.toFixed(6)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-bone-dim">V Spread</span>
            <span className="text-bone">{progress.value_head_spread.toFixed(2)}</span>
          </div>
          <div className="flex justify-between">
            <span className="text-bone-dim">Buffer</span>
            <span className="text-bone">{(progress.buffer_size / 1000).toFixed(0)}k</span>
          </div>
        </div>
      </div>
    </div>
  );
}

// ---------------------------------------------------------------------------
// Training Loss Chart
// ---------------------------------------------------------------------------

function TrainingChart({ history }: { history: TrainingHistoryPoint[] }) {
  if (history.length < 2) {
    return (
      <div className="text-bone-faint text-center py-6 text-sm font-display tracking-widest uppercase">
        Collecting training data...
      </div>
    );
  }

  return (
    <ResponsiveContainer width="100%" height="100%">
      <LineChart data={history} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
        <CartesianGrid strokeDasharray="3 3" stroke="rgba(37,42,74,0.5)" />
        <XAxis
          dataKey="gen"
          tick={{ fill: "#666688", fontSize: 11, fontFamily: "var(--font-mono)" }}
          axisLine={false}
          tickLine={false}
        />
        <YAxis
          yAxisId="loss"
          domain={["auto", "auto"]}
          tick={{ fill: "#666688", fontSize: 11, fontFamily: "var(--font-mono)" }}
          axisLine={false}
          tickLine={false}
          width={48}
          tickFormatter={(v: number) => v.toFixed(2)}
        />
        <YAxis
          yAxisId="rate"
          orientation="right"
          domain={[0, "auto"]}
          tick={{ fill: "#666688", fontSize: 11, fontFamily: "var(--font-mono)" }}
          axisLine={false}
          tickLine={false}
          width={40}
          tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
        />

        <Tooltip
          content={({ active, payload }) => {
            if (!active || !payload?.length) return null;
            const pt = payload[0].payload as TrainingHistoryPoint;
            return (
              <div className="arcane-panel px-4 py-3 text-sm" style={{ background: "#101325f5", border: "1.5px solid #353c6a" }}>
                <div className="text-bone-dim mb-1 font-display text-xs tracking-wider uppercase">Gen {pt.gen}</div>
                <div className="space-y-0.5 font-mono text-[12px]">
                  <div><span className="text-bone-faint">Total: </span><span style={{ color: "#00ddff" }}>{pt.total_loss.toFixed(4)}</span></div>
                  <div><span className="text-bone-faint">Policy: </span><span style={{ color: "#bb66ff" }}>{pt.policy_loss.toFixed(4)}</span></div>
                  <div><span className="text-bone-faint">Value: </span><span style={{ color: "#ff5588" }}>{pt.value_loss.toFixed(4)}</span></div>
                  <div><span className="text-bone-faint">Option: </span><span style={{ color: "#ffaa00" }}>{pt.option_loss.toFixed(4)}</span></div>
                  <div className="mt-1"><span className="text-bone-faint">Win%: </span><span style={{ color: "#00ff88" }}>{(pt.win_rate * 100).toFixed(1)}%</span></div>
                  <div><span className="text-bone-faint">Gen Win%: </span><span style={{ color: "#00ff88" }}>{(pt.gen_win_rate * 100).toFixed(0)}%</span></div>
                </div>
              </div>
            );
          }}
        />

        <Line yAxisId="loss" type="monotone" dataKey="total_loss" stroke="#00ddff" strokeWidth={2} dot={false} isAnimationActive={false} name="Total" />
        <Line yAxisId="loss" type="monotone" dataKey="policy_loss" stroke="#bb66ff" strokeWidth={1.5} dot={false} isAnimationActive={false} name="Policy" />
        <Line yAxisId="loss" type="monotone" dataKey="value_loss" stroke="#ff5588" strokeWidth={1.5} dot={false} isAnimationActive={false} name="Value" />
        <Line yAxisId="loss" type="monotone" dataKey="option_loss" stroke="#ffaa00" strokeWidth={1.5} dot={false} isAnimationActive={false} name="Option" />
        <Line yAxisId="rate" type="monotone" dataKey="win_rate" stroke="#00ff88" strokeWidth={2} strokeDasharray="6 3" dot={false} isAnimationActive={false} name="Win%" />
      </LineChart>
    </ResponsiveContainer>
  );
}

// ---------------------------------------------------------------------------
// Home Page
// ---------------------------------------------------------------------------

export default function Home() {
  const { runs, liveRun: heroRun, liveEvents, connected, loading } = useLocalSocket();
  const { progress: trainingProgress, history: trainingHistory } = useTrainingProgress(15_000);

  const completed = useMemo(() => runs.filter((r) => r.outcome && r.final_floor != null), [runs]);

  const chartData = useMemo(() => {
    const floors = completed.map((r) => r.final_floor ?? 0);
    const avgs = movingAvg(floors, 5);
    return completed.map((r, i) => ({
      index: i + 1,
      runId: r.run_id,
      floor: r.final_floor ?? 0,
      gen: r.gen,
      character: r.character,
      fill: genColor(r.gen),
      avg: avgs[i],
    }));
  }, [completed]);

  const wins = completed.filter((r) => r.outcome === "victory").length;
  const winRate = completed.length > 0 ? `${((wins / completed.length) * 100).toFixed(0)}%` : "—";
  const avgFloor = completed.length > 0
    ? (completed.reduce((s, r) => s + (r.final_floor ?? 0), 0) / completed.length).toFixed(1)
    : "—";
  const bestFloor = completed.length > 0 ? Math.max(...completed.map((r) => r.final_floor ?? 0)) : 0;

  const activeGens = useMemo(() => {
    const gens = new Set(completed.map((r) => r.gen).filter(Boolean));
    return [...gens].sort().map((gen) => [gen, genColor(gen)] as const);
  }, [completed]);

  return (
    <div className="flex gap-3 h-[calc(100vh-16px)]">
      {/* ═══════════════════════════════════════
          Left Sidebar — Live Run + Stats
          ═══════════════════════════════════════ */}
      <div className="w-[300px] shrink-0 flex flex-col gap-2">
        {/* Live Run Panel */}
        {heroRun && (
          <div className="arcane-panel arcane-hero p-3 animate-in flex-1">
            <LiveSidebar run={heroRun} events={liveEvents} connected={connected} />
          </div>
        )}

        {/* Aggregate Stats — 2x2 grid */}
        <div className="grid grid-cols-2 gap-2 animate-in stagger-2">
          <div className="stat-block px-3 py-2.5 text-center">
            <div className="text-3xl font-black font-display text-arcane-bright text-glow-cyan">{completed.length}</div>
            <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mt-0.5">Runs</div>
          </div>
          <div className="stat-block px-3 py-2.5 text-center">
            <div className={`text-3xl font-black font-display ${wins > 0 ? "text-verdant-bright text-glow-green" : "text-bone-dim"}`}>{winRate}</div>
            <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mt-0.5">Win%</div>
          </div>
          <div className="stat-block px-3 py-2.5 text-center">
            <div className="text-3xl font-black font-display text-ember text-glow-amber">{avgFloor}</div>
            <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mt-0.5">Avg Floor</div>
          </div>
          <div className="stat-block px-3 py-2.5 text-center">
            <div className="text-3xl font-black font-display text-mystic text-glow-cyan">{bestFloor || "—"}</div>
            <div className="text-[10px] text-bone-faint uppercase tracking-widest font-display font-bold mt-0.5">Best</div>
          </div>
        </div>

        {/* Logo */}
        <div className="flex items-center justify-center animate-in stagger-3 shrink-0 py-1">
          <img src="/logo.png" alt="Claw the Spire" className="w-full max-w-[220px] object-contain opacity-90" style={{ filter: "drop-shadow(0 0 16px rgba(0,221,255,0.25))" }} />
        </div>

        {/* Training Stats */}
        <TrainingStats progress={trainingProgress} />
      </div>

      {/* ═══════════════════════════════════════
          Right Main — Charts
          ═══════════════════════════════════════ */}
      <div className="flex-1 flex flex-col gap-2 min-w-0">
        {/* Win Expectancy Timeline */}
        <div className="arcane-panel arcane-glow p-3 animate-in stagger-1 flex-[3] min-h-0 flex flex-col">
          <div className="flex items-center justify-between mb-1.5 shrink-0">
            <h2 className="font-display text-sm tracking-[0.15em] uppercase text-arcane-bright font-bold text-glow-cyan">
              Win Expectancy
            </h2>
            <div className="flex gap-4 text-xs text-bone-dim font-display tracking-wider">
              <span className="flex items-center gap-1.5">
                <span className="w-4 h-[3px] rounded inline-block" style={{ background: "#00ddff", boxShadow: "0 0 6px rgba(0,221,255,0.5)" }} /> Value
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-4 h-[3px] inline-block" style={{ borderTop: "2px dashed #ff5588" }} /> HP
              </span>
              <span className="flex items-center gap-1.5">
                <span className="w-3 h-3 inline-block rounded-sm" style={{ background: "rgba(255,34,85,0.15)", border: "1px solid rgba(255,34,85,0.3)" }} /> Combat
              </span>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <TimelineChart events={liveEvents} />
          </div>
        </div>

        {/* Floor Progression */}
        <div className="arcane-panel p-3 animate-in stagger-3 flex-[2] min-h-0 flex flex-col">
          <div className="flex items-center justify-between mb-1.5 shrink-0">
            <h2 className="font-display text-sm tracking-[0.15em] uppercase text-ember-bright font-bold text-glow-amber">
              Floor Progression
            </h2>
            <div className="flex gap-3 text-xs text-bone-dim font-display tracking-wider">
              {activeGens.map(([gen, color]) => (
                <span key={gen} className="flex items-center gap-1">
                  <span className="w-2.5 h-2.5 rounded-sm inline-block" style={{ background: color + "80", boxShadow: `0 0 4px ${color}40` }} /> {gen}
                </span>
              ))}
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <FloorChart data={chartData} bestFloor={bestFloor} />
          </div>
        </div>

        {/* Training Loss Chart */}
        <div className="arcane-panel p-3 animate-in stagger-4 flex-[2] min-h-0 flex flex-col">
          <div className="flex items-center justify-between mb-1.5 shrink-0">
            <h2 className="font-display text-sm tracking-[0.15em] uppercase text-mystic-bright font-bold text-glow-cyan">
              Training Loss
            </h2>
            <div className="flex gap-3 text-xs text-bone-dim font-display tracking-wider">
              <span className="flex items-center gap-1">
                <span className="w-3 h-[2px] rounded inline-block" style={{ background: "#00ddff" }} /> Total
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-[2px] rounded inline-block" style={{ background: "#bb66ff" }} /> Policy
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-[2px] rounded inline-block" style={{ background: "#ff5588" }} /> Value
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-[2px] rounded inline-block" style={{ background: "#ffaa00" }} /> Option
              </span>
              <span className="flex items-center gap-1">
                <span className="w-3 h-[2px] rounded inline-block" style={{ background: "#00ff88", borderTop: "1px dashed #00ff88" }} /> Win%
              </span>
            </div>
          </div>
          <div className="flex-1 min-h-0">
            <TrainingChart history={trainingHistory} />
          </div>
        </div>
      </div>
    </div>
  );
}
