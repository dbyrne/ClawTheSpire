"use client";

import useSWR from "swr";
import { fetcher } from "../lib/api";
import { ExperimentSummary } from "../lib/types";
import ExperimentCard from "../components/ExperimentCard";
import CompactCard from "../components/CompactCard";

type Group = { label: string; items: ExperimentSummary[]; compact: boolean };

function groupByState(exps: ExperimentSummary[]): Group[] {
  const running: ExperimentSummary[] = [];
  const stalled: ExperimentSummary[] = [];
  const stopped: ExperimentSummary[] = [];
  for (const e of exps) {
    const s = e.status?.state ?? "STOPPED";
    if (s === "RUNNING") running.push(e);
    else if (s === "STALLED") stalled.push(e);
    else stopped.push(e);
  }
  const out: Group[] = [];
  if (running.length)
    out.push({ label: "Running", items: running, compact: false });
  if (stalled.length)
    out.push({ label: "Stalled", items: stalled, compact: false });
  if (stopped.length)
    out.push({ label: "Stopped / finalized", items: stopped, compact: true });
  return out;
}

export default function LivePage() {
  const { data, error, isLoading } = useSWR<ExperimentSummary[]>(
    "/api/experiments",
    fetcher,
    { refreshInterval: 10000 },
  );

  return (
    <div>
      <header className="flex items-center justify-between mb-4">
        <h1 className="text-xl font-bold">Experiments</h1>
        <div className="text-xs text-muted mono">
          {data
            ? `${data.filter((e) => e.status.state === "RUNNING").length} running · ${data.length} total`
            : isLoading
              ? "loading…"
              : ""}
        </div>
      </header>

      {error && (
        <div className="bg-bad/10 border border-bad/40 rounded-lg p-3 text-sm text-bad mb-3">
          Failed to load experiments: {String(error.message ?? error)}
        </div>
      )}

      {data &&
        groupByState(data).map((group) => (
          <section key={group.label} className="mb-5">
            <h2 className="text-[11px] uppercase tracking-widest text-muted mb-2 font-semibold">
              {group.label}{" "}
              <span className="text-muted/60">· {group.items.length}</span>
            </h2>
            {group.items.map((e) =>
              group.compact ? (
                <CompactCard key={e.name} exp={e} />
              ) : (
                <ExperimentCard key={e.name} exp={e} />
              ),
            )}
          </section>
        ))}

      {!data && !error && (
        <div className="text-sm text-muted">Loading…</div>
      )}
    </div>
  );
}
