"use client";

import useSWR from "swr";
import { fetcher } from "../lib/api";
import { ExperimentSummary } from "../lib/types";
import ExperimentCard from "../components/ExperimentCard";

type Group = { label: string; items: ExperimentSummary[] };

function groupByState(exps: ExperimentSummary[]): Group[] {
  const buckets: Record<string, ExperimentSummary[]> = {
    RUNNING: [],
    STALLED: [],
    STOPPED: [],
  };
  for (const e of exps) {
    const key = e.status?.state ?? "STOPPED";
    (buckets[key] ?? buckets.STOPPED).push(e);
  }
  const out: Group[] = [];
  if (buckets.RUNNING.length) out.push({ label: "Running", items: buckets.RUNNING });
  if (buckets.STALLED.length) out.push({ label: "Stalled", items: buckets.STALLED });
  if (buckets.STOPPED.length)
    out.push({ label: "Stopped / finalized", items: buckets.STOPPED });
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
          {data ? `${data.length} total` : isLoading ? "loading…" : ""}
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
            <h2 className="text-xs uppercase tracking-wider text-muted mb-2">
              {group.label}
            </h2>
            {group.items.map((e) => (
              <ExperimentCard key={e.name} exp={e} />
            ))}
          </section>
        ))}

      {!data && !error && (
        <div className="text-sm text-muted">Loading…</div>
      )}
    </div>
  );
}
