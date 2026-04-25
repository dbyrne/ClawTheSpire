import { ExperimentSummary } from "../lib/types";
import { formatCost, formatPct } from "../lib/api";
import StatusPill from "./StatusPill";

export default function CompactCard({ exp }: { exp: ExperimentSummary }) {
  const ev = exp.latest_eval;
  const vev = exp.latest_value_eval;
  const mev = exp.latest_mcts_eval;
  const realRescue = mev?.real_rescue_rate ?? mev?.rescue_rate;
  const gen = exp.concluded_gen ?? exp.progress?.gen ?? 0;

  return (
    <section className="bg-panel/80 border border-border/60 rounded-lg px-3 py-2 mb-1.5">
      <div className="flex items-center gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-1.5">
            <span className="text-sm font-medium text-text truncate">
              {exp.name}
            </span>
            {exp.finalized && (
              <span className="text-[9px] text-accent uppercase tracking-wide font-semibold">
                g{exp.concluded_gen}
              </span>
            )}
          </div>
          <div className="text-[10px] text-muted mono truncate">
            {exp.method}
            {exp.params
              ? ` · ${
                  exp.params >= 1_000_000
                    ? `${(exp.params / 1_000_000).toFixed(1)}M`
                    : `${Math.round(exp.params / 1000)}K`
                }`
              : ""}
            {gen ? ` · gen ${gen}` : ""}
          </div>
        </div>
        <div className="flex items-center gap-2 text-[11px] mono text-muted shrink-0">
          {ev?.passed != null && ev?.total && (
            <span>
              <span className="text-muted text-[9px]">P </span>
              <span className="text-text">
                {ev.passed}/{ev.total}
              </span>
            </span>
          )}
          {vev?.passed != null && vev?.total && (
            <span>
              <span className="text-muted text-[9px]">V </span>
              <span className="text-text">
                {vev.passed}/{vev.total}
              </span>
            </span>
          )}
          {exp.best_win_rate != null && (
            <span>
              <span className="text-muted text-[9px]">WR </span>
              <span className="text-text">
                {formatPct(exp.best_win_rate, 0)}
              </span>
            </span>
          )}
          {realRescue != null && (
            <span>
              <span className="text-muted text-[9px]">R </span>
              <span className={realRescue >= 0 ? "text-good" : "text-bad"}>
                {realRescue >= 0 ? "+" : ""}
                {(realRescue * 100).toFixed(0)}%
              </span>
            </span>
          )}
          {exp.worker_cost?.estimated_total_cost != null && (
            <span>
              <span className="text-muted text-[9px]">cost </span>
              <span className="text-text">
                {formatCost(exp.worker_cost.estimated_total_cost)}
              </span>
            </span>
          )}
        </div>
        <StatusPill state={exp.status.state} />
      </div>
    </section>
  );
}
