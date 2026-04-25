import { DistillSummary } from "../lib/types";
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
      {hint && <div className="text-[10px] text-muted mono">{hint}</div>}
    </div>
  );
}

export default function DistillCard({ exp }: { exp: DistillSummary }) {
  const epoch = exp.current_epoch ?? exp.concluded_epoch ?? 0;
  const total = exp.epochs_total ?? "?";
  const pct =
    typeof total === "number" && total > 0
      ? Math.min(100, Math.round((epoch / total) * 100))
      : 0;
  const ev = exp.latest_eval;
  const vev = exp.latest_value_eval;
  const mev = exp.latest_mcts_eval;
  const realRescue = mev?.real_rescue_rate ?? mev?.rescue_rate;

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
                done e{exp.concluded_epoch}
              </span>
            )}
          </div>
          <div className="text-xs text-muted mono">
            {exp.method}
            {exp.dataset ? ` · ${exp.dataset}` : ""}
          </div>
        </div>
        <StatusPill state={exp.status.state} />
      </div>

      <div className="mb-3">
        <div className="flex justify-between text-xs mono text-muted mb-1">
          <span>
            epoch {epoch}/{total}
          </span>
          <span>
            {exp.status.age_s != null
              ? `${formatDuration(exp.status.age_s)} ago`
              : "-"}
            {exp.epoch_time_last != null
              ? ` · ${formatDuration(exp.epoch_time_last)}/epoch`
              : ""}
          </span>
        </div>
        <div className="h-1.5 bg-panel2 rounded overflow-hidden">
          <div
            className={`h-full transition-all ${
              exp.status.state === "RUNNING"
                ? "bg-good"
                : exp.status.state === "STALLED"
                ? "bg-warn"
                : "bg-muted"
            }`}
            style={{ width: `${pct}%` }}
          />
        </div>
      </div>

      <div className="grid grid-cols-3 gap-x-3 gap-y-3 mb-3">
        <Metric
          label="val top1"
          value={formatPct(exp.val_top1_last10)}
          hint={
            exp.val_top1_best != null
              ? `peak ${formatPct(exp.val_top1_best)}`
              : undefined
          }
        />
        <Metric
          label="val pol-loss"
          value={formatNum(exp.val_pol_loss_last10, 3)}
        />
        <Metric
          label="val v-loss"
          value={formatNum(exp.val_val_loss_last10, 3)}
        />
      </div>

      <div className="grid grid-cols-4 gap-x-3 gap-y-2 pt-2 border-t border-border">
        <Metric
          label="P-Eval"
          value={
            ev?.passed != null && ev?.total
              ? `${ev.passed}/${ev.total}`
              : "-"
          }
          hint={ev?.gen != null ? `g${ev.gen}` : undefined}
        />
        <Metric
          label="V-Eval"
          value={
            vev?.passed != null && vev?.total
              ? `${vev.passed}/${vev.total}`
              : "-"
          }
          hint={vev?.gen != null ? `g${vev.gen}` : undefined}
        />
        <Metric
          label="real rescue"
          value={
            realRescue != null
              ? `${realRescue >= 0 ? "+" : ""}${(realRescue * 100).toFixed(0)}%`
              : "-"
          }
        />
        <Metric
          label="pol-only WR"
          value={formatPct(exp.policy_only_wr)}
        />
      </div>
    </section>
  );
}
