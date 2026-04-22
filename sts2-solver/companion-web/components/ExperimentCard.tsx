import { ExperimentSummary } from "../lib/types";
import { formatDuration, formatNum, formatPct } from "../lib/api";
import StatusPill from "./StatusPill";

function Metric({
  label,
  value,
  hint,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
}) {
  return (
    <div className="flex flex-col">
      <div className="text-[10px] uppercase tracking-wide text-muted">
        {label}
      </div>
      <div className="mono text-sm text-text">{value}</div>
      {hint && (
        <div className="text-[10px] text-muted mono">{hint}</div>
      )}
    </div>
  );
}

export default function ExperimentCard({ exp }: { exp: ExperimentSummary }) {
  const { progress, status } = exp;
  const gen = progress?.gen ?? exp.concluded_gen ?? 0;
  const total = exp.generations_total ?? "?";
  const phase = progress?.phase;
  const pct =
    typeof total === "number" && total > 0
      ? Math.min(100, Math.round((gen / total) * 100))
      : 0;

  return (
    <section className="bg-panel border border-border rounded-xl p-4 mb-3">
      <div className="flex items-start gap-2 mb-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-base font-semibold text-text truncate">
              {exp.name}
            </h2>
            {exp.finalized && (
              <span className="text-[10px] text-accent border border-accent/40 rounded px-1.5 py-0.5 uppercase tracking-wide">
                finalized g{exp.concluded_gen}
              </span>
            )}
          </div>
          <div className="text-xs text-muted mono">
            {exp.method}
            {exp.parent ? ` · from ${exp.parent}` : ""}
            {exp.encounter_set ? ` · ${exp.encounter_set}` : ""}
          </div>
        </div>
        <StatusPill
          state={status.state}
          label={phase && status.state === "RUNNING" ? phase : undefined}
        />
      </div>

      <div className="mb-3">
        <div className="flex justify-between text-xs mono text-muted mb-1">
          <span>
            gen {gen}/{total}
          </span>
          <span>
            {status.age_s != null ? `${formatDuration(status.age_s)} ago` : "-"}
            {status.cadence_s != null
              ? ` · ${formatDuration(status.cadence_s)}/gen`
              : ""}
          </span>
        </div>
        <div className="h-1.5 bg-panel2 rounded overflow-hidden">
          <div
            className={`h-full transition-all ${
              status.state === "RUNNING"
                ? "bg-good"
                : status.state === "STALLED"
                ? "bg-warn"
                : "bg-muted"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-x-3 gap-y-3 mb-3">
        <Metric
          label="WR (last 10)"
          value={formatPct(exp.win_rate_last10)}
          hint={
            progress?.best_win_rate != null
              ? `peak ${formatPct(progress.best_win_rate)}`
              : undefined
          }
        />
        <Metric
          label="P-loss"
          value={formatNum(exp.policy_loss_last10, 3)}
        />
        <Metric
          label="V-loss"
          value={formatNum(exp.value_loss_last10, 3)}
        />
      </div>

      {(exp.kl_mcts_net_last10 != null ||
        exp.top1_agree_last10 != null ||
        exp.value_corr_last10 != null) && (
        <div className="grid grid-cols-3 gap-x-3 gap-y-2 pt-2 border-t border-border">
          <Metric
            label="KL(mcts‖net)"
            value={formatNum(exp.kl_mcts_net_last10, 3)}
            hint="echo if →0"
          />
          <Metric
            label="top1-agree"
            value={formatNum(exp.top1_agree_last10, 2)}
            hint="echo if ≥.90"
          />
          <Metric
            label="V corr"
            value={formatNum(exp.value_corr_last10, 2)}
          />
        </div>
      )}
    </section>
  );
}
