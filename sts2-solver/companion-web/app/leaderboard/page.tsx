"use client";

import { useState } from "react";
import useSWR from "swr";
import { fetcher, formatPct } from "../../lib/api";
import { Leaderboard, LeaderboardEntry } from "../../lib/types";

function EntryRow({ e, rank }: { e: LeaderboardEntry; rank: number }) {
  return (
    <div className="flex items-center gap-2 py-1.5 border-b border-border/40 last:border-0 text-sm">
      <div className="w-5 text-muted text-xs mono text-right">{rank}</div>
      <div className="flex-1 min-w-0 truncate">{e.experiment}</div>
      <div className="mono text-xs text-muted">g{e.gen ?? "?"}</div>
      <div className="mono w-12 text-right">
        {e.passed}/{e.total}
      </div>
      <div className="mono w-14 text-right text-accent">
        {formatPct(e.score)}
      </div>
    </div>
  );
}

function CategoryCard({
  cat,
  entries,
}: {
  cat: string;
  entries: LeaderboardEntry[];
}) {
  const top = entries.slice(0, 5);
  return (
    <section className="bg-panel border border-border rounded-xl p-3 mb-3">
      <h3 className="text-sm font-semibold mb-1">{cat}</h3>
      <div className="text-[10px] text-muted mono uppercase tracking-wider mb-1.5">
        best across {entries.length} eval run{entries.length === 1 ? "" : "s"}
      </div>
      {top.map((e, i) => (
        <EntryRow key={i} e={e} rank={i + 1} />
      ))}
    </section>
  );
}

export default function LeaderboardPage() {
  const { data, error, isLoading } = useSWR<Leaderboard>(
    "/api/leaderboard",
    fetcher,
    { refreshInterval: 60000 },
  );

  const [tab, setTab] = useState<"policy_eval" | "value_eval">("policy_eval");

  if (error)
    return (
      <div className="bg-bad/10 border border-bad/40 rounded-lg p-3 text-sm text-bad">
        Failed to load: {String(error.message ?? error)}
      </div>
    );
  if (isLoading || !data)
    return <div className="text-sm text-muted">Loading…</div>;

  const section = data[tab];
  const categories = Object.entries(section.by_category).sort(
    ([a], [b]) => a.localeCompare(b),
  );

  return (
    <div>
      <header className="mb-3">
        <h1 className="text-xl font-bold mb-3">Eval leaderboard</h1>
        <div className="inline-flex bg-panel border border-border rounded-full p-0.5">
          {(["policy_eval", "value_eval"] as const).map((k) => (
            <button
              key={k}
              onClick={() => setTab(k)}
              className={`px-4 py-1.5 text-xs rounded-full font-medium transition-colors ${
                tab === k
                  ? "bg-accent text-white"
                  : "text-muted hover:text-text"
              }`}
            >
              {k === "policy_eval" ? "P-Eval" : "V-Eval"}
            </button>
          ))}
        </div>
      </header>

      <section className="bg-panel border border-border rounded-xl p-3 mb-4">
        <h3 className="text-sm font-semibold mb-1">Overall best</h3>
        <div className="text-[10px] text-muted mono uppercase tracking-wider mb-1.5">
          by total pass rate, across all experiments × gens
        </div>
        {section.overall_top.slice(0, 10).map((e, i) => (
          <EntryRow key={i} e={e} rank={i + 1} />
        ))}
      </section>

      <h2 className="text-xs uppercase tracking-wider text-muted mb-2">
        Best by category
      </h2>
      {categories.map(([cat, entries]) => (
        <CategoryCard key={cat} cat={cat} entries={entries} />
      ))}
    </div>
  );
}
