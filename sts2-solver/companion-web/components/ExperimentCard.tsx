import { ExperimentSummary } from "../lib/types";
import { formatDuration, formatNum, formatPct } from "../lib/api";
import StatusPill from "./StatusPill";

function Metric({
  label,
  value,
  hint,
  delta,
  deltaFormat,
  higherIsBetter = true,
}: {
  label: string;
  value: React.ReactNode;
  hint?: string;
  delta?: number | null;
  deltaFormat?: (d: number) => string;
  higherIsBetter?: boolean;
}) {
  let deltaEl: React.ReactNode = null;
  if (delta != null && deltaFormat) {
    const eps = 1e-9;
    const dir =
      delta > eps ? 1 : delta < -eps ? -1 : 0;
    const color =
      dir === 0
        ? "text-muted"
        : higherIsBetter
          ? dir > 0
            ? "text-good"
            : "text-bad"
          : dir > 0
            ? "text-bad"
            : "text-good";
    const sign = delta >= 0 ? "+" : "";
    const arrow = dir > 0 ? "↑" : dir < 0 ? "↓" : "·";
    deltaEl = (
      <span className={`${color} text-[10px] mono ml-1`}>
        {arrow} {sign}{deltaFormat(delta)}
      </span>
    );
  }
  return (
    <div className="flex flex-col">
      <div className="text-[10px] uppercase tracking-wide text-muted">
        {label}
      </div>
      <div className="mono text-sm text-text flex items-baseline">
        <span>{value}</span>
        {deltaEl}
      </div>
      {hint && (
        <div className="text-[10px] text-muted mono">{hint}</div>
      )}
    </div>
  );
}

export default function ExperimentCard({ exp }: { exp: ExperimentSummary }) {
  const { progress, status } = exp;
  const ev = exp.latest_eval;
  const vev = exp.latest_value_eval;
  const mev = exp.latest_mcts_eval;
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
            {exp.params != null
              ? ` · ${Math.round(exp.params / 1000)}K params`
              : ""}
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
        {(exp.training_elapsed_s != null || exp.training_eta_s != null) && (
          <div className="flex justify-between text-[10px] mono text-muted mt-1">
            <span>
              {exp.training_elapsed_s != null
                ? `trained ${formatDuration(exp.training_elapsed_s)}`
                : ""}
            </span>
            <span>
              {exp.training_eta_s != null
                ? `ETA ${formatDuration(exp.training_eta_s)}`
                : ""}
            </span>
          </div>
        )}
      </div>

      <div className="grid grid-cols-4 gap-x-3 gap-y-3 mb-3">
        <Metric
          label="P-Eval"
          value={
            ev?.passed != null && ev?.total
              ? `${ev.passed}/${ev.total}`
              : "-"
          }
          hint={
            exp.peak_eval?.passed != null && exp.peak_eval?.total
              ? `peak ${exp.peak_eval.passed}/${exp.peak_eval.total}${
                  exp.peak_eval.gen != null ? ` · g${exp.peak_eval.gen}` : ""
                }`
              : undefined
          }
          delta={exp.eval_delta}
          deltaFormat={(d) => `${(d * 100).toFixed(1)}%`}
        />
        <Metric
          label="V-Eval"
          value={
            vev?.passed != null && vev?.total
              ? `${vev.passed}/${vev.total}`
              : "-"
          }
          hint={
            exp.peak_value_eval?.passed != null && exp.peak_value_eval?.total
              ? `peak ${exp.peak_value_eval.passed}/${exp.peak_value_eval.total}${
                  exp.peak_value_eval.gen != null
                    ? ` · g${exp.peak_value_eval.gen}`
                    : ""
                }`
              : undefined
          }
          delta={exp.value_eval_delta}
          deltaFormat={(d) => `${(d * 100).toFixed(1)}%`}
        />
        <Metric
          label="rescue"
          value={
            mev?.rescue_rate != null
              ? `${mev.rescue_rate >= 0 ? "+" : ""}${(mev.rescue_rate * 100).toFixed(0)}%`
              : "-"
          }
          hint={
            mev?.clean != null
              ? `CL${mev.clean} EC${mev.echo ?? 0} FX${mev.fixed ?? 0}`
              : undefined
          }
          delta={exp.rescue_delta}
          deltaFormat={(d) => `${(d * 100).toFixed(0)}%`}
        />
        <Metric
          label="WR (last 10)"
          value={formatPct(exp.win_rate_last10)}
          hint={
            exp.best_win_rate != null
              ? `peak ${formatPct(exp.best_win_rate)}${
                  exp.best_win_rate_gen != null
                    ? ` · g${exp.best_win_rate_gen}`
                    : ""
                }`
              : undefined
          }
          delta={exp.win_rate_delta}
          deltaFormat={(d) => `${(d * 100).toFixed(1)}%`}
        />
      </div>

      <div className="grid grid-cols-3 gap-x-3 gap-y-3 mb-3 pt-3 border-t border-border">
        <Metric
          label="P-loss"
          value={formatNum(exp.policy_loss_last10, 3)}
          delta={exp.policy_loss_delta}
          deltaFormat={(d) => d.toFixed(3)}
          higherIsBetter={false}
        />
        <Metric
          label="V-loss"
          value={formatNum(exp.value_loss_last10, 3)}
          delta={exp.value_loss_delta}
          deltaFormat={(d) => d.toFixed(3)}
          higherIsBetter={false}
        />
        <Metric
          label="gen-time"
          value={
            exp.gen_time_last != null
              ? formatDuration(exp.gen_time_last)
              : "-"
          }
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
            delta={exp.kl_mcts_net_delta}
            deltaFormat={(d) => d.toFixed(3)}
            higherIsBetter={false}
          />
          <Metric
            label="top1-agree"
            value={formatNum(exp.top1_agree_last10, 2)}
            hint="echo if ≥.90"
            delta={exp.top1_agree_delta}
            deltaFormat={(d) => `${(d * 100).toFixed(1)}%`}
            higherIsBetter={false}
          />
          <Metric
            label="V corr"
            value={formatNum(exp.value_corr_last10, 2)}
            delta={exp.value_corr_delta}
            deltaFormat={(d) => d.toFixed(3)}
          />
        </div>
      )}
    </section>
  );
}
