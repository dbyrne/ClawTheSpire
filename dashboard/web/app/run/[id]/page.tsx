"use client";

import { useMemo, use } from "react";
import Link from "next/link";
import { useRunDetail } from "@/lib/useLocalSocket";
import type { Run, RunEvent } from "@/lib/types";
import {
  AreaChart,
  Area,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  ResponsiveContainer,
  ReferenceLine,
  ReferenceArea,
} from "recharts";

// ---------------------------------------------------------------------------
// Data transforms
// ---------------------------------------------------------------------------

interface ChartPoint {
  index: number;
  floor: number;
  hp: number | null;
  networkValue: number | null;
  label: string;
  eventType: string;
  detail: string;
}

function buildChartData(events: RunEvent[]): ChartPoint[] {
  const points: ChartPoint[] = [];
  let idx = 0;
  for (const ev of events) {
    const floor = ev.floor ?? 0;
    const hp = ev.hp;
    const nv = ev.network_value;
    const detail = ev.detail as Record<string, unknown> | null;
    let label = "", detailStr = "";
    switch (ev.event_type) {
      case "combat_start": label = detailStr = `Combat: ${(detail?.enemies as string[])?.join(", ") ?? "?"}`; break;
      case "combat_turn": label = detailStr = `T${detail?.turn ?? "?"}: ${(detail?.cards_played as string[])?.join(", ") ?? ""}`; break;
      case "combat_end": label = detailStr = `Combat ${detail?.outcome ?? "?"} (${detail?.turns ?? "?"} turns)`; break;
      case "decision": label = detailStr = `${detail?.screen_type ?? "?"}: ${detail?.choice ?? ""}`; break;
      case "run_start": label = detailStr = "Run start"; break;
      case "run_end": label = detailStr = `Run ${detail?.outcome ?? "?"}`; break;
      default: label = ev.event_type; detailStr = JSON.stringify(detail);
    }
    points.push({ index: idx++, floor, hp, networkValue: nv != null ? Math.round(nv * 100) / 100 : null, label, eventType: ev.event_type, detail: detailStr });
  }
  return points;
}

function findCombatRegions(points: ChartPoint[]): { start: number; end: number; enemies: string }[] {
  const regions: { start: number; end: number; enemies: string }[] = [];
  let combatStart: number | null = null;
  let enemies = "";
  for (const pt of points) {
    if (pt.eventType === "combat_start") { combatStart = pt.index; enemies = pt.detail.replace("Combat: ", ""); }
    else if (pt.eventType === "combat_end" && combatStart != null) { regions.push({ start: combatStart, end: pt.index, enemies }); combatStart = null; }
  }
  if (combatStart != null) regions.push({ start: combatStart, end: points.length - 1, enemies });
  return regions;
}

function toWinPct(nv: number): number {
  return ((nv + 1) / 2) * 100;
}

// ---------------------------------------------------------------------------
// Event row color
// ---------------------------------------------------------------------------

function eventRowClass(eventType: string): string {
  switch (eventType) {
    case "combat_start": return "border-l-2 border-l-blood/40 bg-blood/[0.03]";
    case "combat_end": return "border-l-2 border-l-verdant/40 bg-verdant/[0.03]";
    case "combat_turn": return "border-l-2 border-l-rune-light/30";
    case "decision": return "border-l-2 border-l-ember/30 bg-ember/[0.02]";
    case "run_end": return "border-l-2 border-l-ember-bright/50 bg-ember-bright/[0.03]";
    default: return "border-l-2 border-l-transparent";
  }
}

// ---------------------------------------------------------------------------
// Page
// ---------------------------------------------------------------------------

export default function RunPage({ params }: { params: Promise<{ id: string }> }) {
  const { id } = use(params);
  const { run, events, loading } = useRunDetail(id);

  const chartData = useMemo(() => buildChartData(events), [events]);
  const combatRegions = useMemo(() => findCombatRegions(chartData), [chartData]);

  if (loading) {
    return <div className="text-bone-dim text-center py-20 font-serif text-lg italic">Loading...</div>;
  }
  if (!run) {
    return <div className="text-bone-dim text-center py-20 font-serif text-lg italic">Run not found in the archive</div>;
  }

  const hasValues = chartData.some((p) => p.networkValue != null);
  const isLive = !run.outcome;

  const combats = events.filter((e) => e.event_type === "combat_end");
  const combatsWon = combats.filter((e) => (e.detail as Record<string, unknown>)?.outcome === "win").length;
  const totalTurns = combats.reduce((s, e) => s + ((e.detail as Record<string, unknown>)?.turns as number ?? 0), 0);
  const decisions = events.filter((e) => e.event_type === "decision").length;
  const valuePoints = chartData.filter((p) => p.networkValue != null);
  const avgValue = valuePoints.length > 0 ? valuePoints.reduce((s, p) => s + (p.networkValue ?? 0), 0) / valuePoints.length : null;

  const eventLog = events
    .filter((e) => e.event_type !== "run_start")
    .map((e) => {
      const detail = e.detail as Record<string, unknown> | null;
      let text = e.event_type;
      if (e.event_type === "combat_start") text = `Combat: ${(detail?.enemies as string[])?.join(", ") ?? "?"}`;
      else if (e.event_type === "combat_turn") text = `T${detail?.turn}: ${(detail?.cards_played as string[])?.join(", ") ?? "-"}`;
      else if (e.event_type === "combat_end") text = `${detail?.outcome} (${detail?.turns} turns)`;
      else if (e.event_type === "decision") text = `${detail?.screen_type}: ${detail?.choice ?? ""}`;
      else if (e.event_type === "run_end") text = `Run ${detail?.outcome}`;
      const hs = detail?.head_scores as { head: string; chosen: number; options: { label: string; score: number }[] } | undefined;
      return { ...e, text, nv: e.network_value != null ? `${toWinPct(e.network_value).toFixed(0)}%` : null, hs };
    });

  return (
    <div>
      {/* Back link */}
      <Link href="/" className="text-ember-dim hover:text-ember transition-colors text-xs font-display tracking-wider uppercase mb-5 inline-block">
        &larr; Back to Oracle
      </Link>

      {/* ── Header ── */}
      <div className="arcane-panel arcane-hero p-5 mb-5 animate-in">
        <div className="flex items-start justify-between">
          <div>
            <div className="flex items-center gap-3 mb-2">
              {isLive && <div className="live-dot" />}
              <h1 className="font-display text-xl tracking-[0.1em] uppercase text-bone">
                Run {run.run_id.slice(0, 12)}
              </h1>
              {run.outcome && (
                <span className={`font-mono text-xs font-bold uppercase tracking-wider px-2 py-0.5 rounded-sm ${
                  run.outcome === "victory"
                    ? "text-ember-bright bg-ember-bright/10 border border-ember-bright/20"
                    : "text-blood-bright bg-blood-bright/10 border border-blood-bright/20"
                }`}>
                  {run.outcome}
                </span>
              )}
            </div>
            <div className="flex items-center gap-4 text-sm">
              <span className={`font-serif text-lg font-semibold ${run.character === "The Ironclad" ? "text-blood-bright" : "text-verdant-bright"}`}>
                {run.character}
              </span>
              <span className="text-bone-faint text-xs font-mono">{run.checkpoint}</span>
              <span className="text-bone-faint text-xs font-mono">{run.gen}</span>
            </div>
          </div>

          {/* Run stats row */}
          <div className="flex gap-6">
            {[
              { label: "Floor", value: run.final_floor ?? "?", accent: "text-ember" },
              { label: "HP", value: `${run.final_hp ?? "?"}/${run.max_hp ?? "?"}`, accent: "text-blood-bright" },
              { label: "Combats", value: `${combatsWon}/${combats.length}`, accent: "text-bone" },
              { label: "Turns", value: totalTurns, accent: "text-bone" },
              ...(avgValue != null ? [{ label: "Avg Value", value: `${toWinPct(avgValue).toFixed(0)}%`, accent: "text-arcane" }] : []),
            ].map((stat) => (
              <div key={stat.label} className="text-right">
                <div className="text-xs text-bone-dim uppercase tracking-[0.12em] font-display">{stat.label}</div>
                <div className={`text-xl font-bold font-mono ${stat.accent}`}>{stat.value}</div>
              </div>
            ))}
          </div>
        </div>
      </div>

      {/* ── Win Expectancy Chart ── */}
      <div className="arcane-panel arcane-glow p-4 mb-5 animate-in stagger-1">
        <div className="flex items-center justify-between mb-3">
          <h2 className="font-display text-base tracking-[0.12em] uppercase text-bone">
            Win Expectancy
            {!hasValues && <span className="text-bone-faint text-xs ml-3 font-mono normal-case">(values appear with new runs)</span>}
          </h2>
          <div className="flex gap-5 text-xs text-bone-dim">
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-[2px] bg-arcane rounded inline-block" /> Network Value
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-3 h-[2px] inline-block" style={{ borderTop: "1.5px dashed #b91c1c" }} /> HP
            </span>
            <span className="flex items-center gap-1.5">
              <span className="w-2.5 h-2.5 inline-block rounded-sm" style={{ background: "rgba(185,28,28,0.12)" }} /> Combat
            </span>
          </div>
        </div>
        <div className="h-72">
          <ResponsiveContainer width="100%" height="100%">
            <AreaChart data={chartData} margin={{ top: 8, right: 16, bottom: 4, left: 0 }}>
              <defs>
                <linearGradient id="detailValueGrad" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#3b82f6" stopOpacity={0.3} />
                  <stop offset="45%" stopColor="#3b82f6" stopOpacity={0.04} />
                  <stop offset="55%" stopColor="#b91c1c" stopOpacity={0.03} />
                  <stop offset="100%" stopColor="#b91c1c" stopOpacity={0.12} />
                </linearGradient>
              </defs>

              <CartesianGrid strokeDasharray="3 3" stroke="rgba(28,22,54,0.4)" />
              <XAxis dataKey="index" tick={{ fill: "#524a3a", fontSize: 12 }} axisLine={{ stroke: "#1c1636" }} tickLine={false} />
              <YAxis yAxisId="value" domain={[-1, 1]} tick={{ fill: "#524a3a", fontSize: 12 }}
                tickFormatter={(v: number) => `${toWinPct(v).toFixed(0)}%`} axisLine={false} tickLine={false} width={44} />
              <YAxis yAxisId="hp" orientation="right" domain={[0, "dataMax"]} tick={{ fill: "#524a3a", fontSize: 12 }}
                axisLine={false} tickLine={false} width={36} />

              <Tooltip content={({ active, payload }) => {
                if (!active || !payload?.length) return null;
                const pt = payload[0].payload as ChartPoint;
                return (
                  <div className="arcane-panel px-3 py-2 text-xs" style={{ background: "#0d0d1cf0", border: "1px solid #2d2350" }}>
                    <div className="text-bone-dim">Floor {pt.floor}</div>
                    <div className="text-bone mt-0.5 mb-1">{pt.label}</div>
                    {pt.networkValue != null && <div className="text-arcane font-bold">{toWinPct(pt.networkValue).toFixed(0)}% win exp</div>}
                    {pt.hp != null && <div className="text-blood-bright">HP {pt.hp}</div>}
                  </div>
                );
              }} />

              {combatRegions.map((r, i) => (
                <ReferenceArea key={i} yAxisId="value" x1={r.start} x2={r.end} fill="rgba(185,28,28,0.07)" strokeOpacity={0} />
              ))}

              <ReferenceLine yAxisId="value" y={0} stroke="#2d2350" strokeDasharray="4 4" />

              <Area yAxisId="value" type="monotone" dataKey="networkValue" stroke="#3b82f6" strokeWidth={2}
                fill="url(#detailValueGrad)" connectNulls isAnimationActive={false} dot={false} />
              <Line yAxisId="hp" type="stepAfter" dataKey="hp" stroke="#b91c1c" strokeWidth={1.5}
                strokeDasharray="6 3" dot={false} connectNulls isAnimationActive={false} />
            </AreaChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* ── Event Timeline ── */}
      <div className="arcane-panel overflow-hidden animate-in stagger-2">
        <div className="px-4 py-2.5 border-b border-rune flex items-center justify-between">
          <h2 className="font-display text-base tracking-[0.12em] uppercase text-bone">Event Timeline</h2>
          <span className="text-xs text-bone-faint font-mono">{eventLog.length} events &middot; {decisions} decisions</span>
        </div>
        <div className="max-h-[460px] overflow-y-auto">
          <table className="w-full text-sm">
            <thead>
              <tr className="text-xs text-bone-dim uppercase tracking-wider border-b border-rune sticky top-0 bg-obsidian">
                <th className="px-4 py-2 text-left w-14">Floor</th>
                <th className="px-4 py-2 text-right w-14">HP</th>
                <th className="px-4 py-2 text-right w-16">Value</th>
                <th className="px-4 py-2 text-left">Event</th>
              </tr>
            </thead>
            <tbody>
              {eventLog.map((e, i) => {
                const hasScores = e.hs?.options?.length;
                const maxScore = hasScores ? Math.max(...e.hs!.options.map((o) => Math.abs(o.score)), 0.01) : 0;
                return (
                  <tr key={e.id ?? i} className={`rune-row ${eventRowClass(e.event_type)}`}>
                    <td className="px-4 py-1 text-bone-dim align-top">{e.floor ?? "—"}</td>
                    <td className="px-4 py-1 text-right text-blood align-top">{e.hp ?? "—"}</td>
                    <td className="px-4 py-1 text-right text-arcane font-mono align-top">{e.nv ?? <span className="text-bone-faint">—</span>}</td>
                    <td className="px-4 py-1 text-bone">
                      <div className="truncate max-w-lg">{e.text}</div>
                      {hasScores && (
                        <div className="mt-1 mb-0.5 flex flex-wrap gap-x-4 gap-y-0.5 text-xs">
                          {e.hs!.options
                            .map((opt, oi) => ({ ...opt, oi }))
                            .sort((a, b) => b.score - a.score)
                            .slice(0, 8)
                            .map((opt) => {
                              const isChosen = opt.oi === e.hs!.chosen;
                              const barW = Math.max(6, (Math.abs(opt.score) / maxScore) * 100);
                              return (
                                <span key={opt.oi} className={`flex items-center gap-1.5 ${isChosen ? "text-bone" : "text-bone-faint"}`}>
                                  <span className="font-mono text-[11px] w-[80px] truncate" title={opt.label}>
                                    {isChosen ? "▸" : " "}{opt.label}
                                  </span>
                                  <span className="w-[50px] h-[5px] bg-rune/40 rounded-sm overflow-hidden inline-block">
                                    <span
                                      className="block h-full rounded-sm"
                                      style={{
                                        width: `${barW}%`,
                                        background: isChosen ? "#60a5fa" : "#3d3570",
                                      }}
                                    />
                                  </span>
                                  <span className="font-mono text-[11px] w-[34px] text-right">{opt.score.toFixed(2)}</span>
                                </span>
                              );
                            })}
                        </div>
                      )}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
