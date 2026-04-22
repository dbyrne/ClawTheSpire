"use client";

import { useMemo, useState } from "react";
import useSWR from "swr";
import { fetcher, formatPct } from "../../lib/api";
import { BenchmarkRow } from "../../lib/types";

type SortKey = "win_rate" | "games" | "gen" | "experiment";

export default function BenchmarksPage() {
  const { data, error, isLoading } = useSWR<BenchmarkRow[]>(
    "/api/benchmarks",
    fetcher,
    { refreshInterval: 60000 },
  );

  const [filter, setFilter] = useState("");
  const [sortKey, setSortKey] = useState<SortKey>("win_rate");
  const [sortDesc, setSortDesc] = useState(true);

  const rows = useMemo(() => {
    if (!data) return [];
    const f = filter.toLowerCase();
    const filtered = f
      ? data.filter(
          (r) =>
            r.experiment?.toLowerCase().includes(f) ||
            r.suite?.toLowerCase().includes(f) ||
            r.encounter_set?.toLowerCase().includes(f) ||
            r.mode?.toLowerCase().includes(f),
        )
      : data;
    const sorted = [...filtered].sort((a, b) => {
      const av = (a[sortKey] as number | string | undefined) ?? "";
      const bv = (b[sortKey] as number | string | undefined) ?? "";
      if (av === bv) return 0;
      if (av === "") return 1;
      if (bv === "") return -1;
      return sortDesc ? (bv > av ? 1 : -1) : av > bv ? 1 : -1;
    });
    return sorted;
  }, [data, filter, sortKey, sortDesc]);

  const toggle = (key: SortKey) => {
    if (key === sortKey) setSortDesc((d) => !d);
    else {
      setSortKey(key);
      setSortDesc(true);
    }
  };

  const H = ({ k, label }: { k: SortKey; label: string }) => (
    <th
      onClick={() => toggle(k)}
      className="px-2 py-1.5 text-left font-medium text-muted uppercase text-[10px] tracking-wider cursor-pointer select-none"
    >
      {label} {sortKey === k ? (sortDesc ? "↓" : "↑") : ""}
    </th>
  );

  return (
    <div>
      <header className="mb-3">
        <h1 className="text-xl font-bold mb-2">Benchmarks</h1>
        <input
          type="search"
          placeholder="filter by experiment / suite / encounter set"
          value={filter}
          onChange={(e) => setFilter(e.target.value)}
          className="w-full bg-panel border border-border rounded px-3 py-2 text-sm placeholder:text-muted focus:outline-none focus:border-accent"
        />
      </header>

      {error && (
        <div className="bg-bad/10 border border-bad/40 rounded-lg p-3 text-sm text-bad mb-3">
          Failed to load: {String(error.message ?? error)}
        </div>
      )}
      {isLoading && <div className="text-sm text-muted">Loading…</div>}

      {rows && rows.length > 0 && (
        <div className="bg-panel border border-border rounded-xl overflow-x-auto">
          <table className="w-full text-sm mono">
            <thead className="bg-panel2 border-b border-border sticky top-0">
              <tr>
                <H k="experiment" label="Exp" />
                <H k="gen" label="Gen" />
                <H k="win_rate" label="WR" />
                <th className="px-2 py-1.5 text-left font-medium text-muted uppercase text-[10px] tracking-wider">
                  CI
                </th>
                <H k="games" label="N" />
                <th className="px-2 py-1.5 text-left font-medium text-muted uppercase text-[10px] tracking-wider">
                  Suite
                </th>
                <th className="px-2 py-1.5 text-left font-medium text-muted uppercase text-[10px] tracking-wider">
                  Sims
                </th>
              </tr>
            </thead>
            <tbody>
              {rows.map((r, i) => (
                <tr
                  key={i}
                  className="border-b border-border/60 last:border-0"
                >
                  <td className="px-2 py-1.5 truncate max-w-[140px]">
                    {r.experiment}
                  </td>
                  <td className="px-2 py-1.5">{r.gen ?? "-"}</td>
                  <td className="px-2 py-1.5">
                    {r.win_rate != null ? formatPct(r.win_rate) : "-"}
                  </td>
                  <td className="px-2 py-1.5 text-muted text-xs">
                    {r.ci_low != null && r.ci_high != null
                      ? `${(r.ci_low * 100).toFixed(1)}–${(r.ci_high * 100).toFixed(1)}`
                      : "-"}
                  </td>
                  <td className="px-2 py-1.5">{r.games ?? "-"}</td>
                  <td className="px-2 py-1.5 truncate max-w-[120px] text-muted text-xs">
                    {r.encounter_set ?? r.suite ?? "-"}
                  </td>
                  <td className="px-2 py-1.5 text-xs text-muted">
                    {r.mcts_sims ?? "-"}
                    {r.pomcp ? "p" : ""}
                  </td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {data && rows.length === 0 && (
        <div className="text-sm text-muted">No benchmark rows match.</div>
      )}
    </div>
  );
}
