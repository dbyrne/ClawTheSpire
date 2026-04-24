import { ExperimentSummary } from "../lib/types";
import { formatDuration, formatNum, formatPct } from "../lib/api";
import StatusPill from "./StatusPill";
import MetricCell from "./MetricCell";
import Sparkline from "./Sparkline";

function MiniStat({
  label,
  value,
  delta,
  deltaFormat,
  higherIsBetter = true,
  series = [],
}: {
  label: string;
  value: React.ReactNode;
  delta?: number | null;
  deltaFormat?: (d: number) => string;
  higherIsBetter?: boolean;
  series?: Parameters<typeof Sparkline>[0]["points"];
}) {
  let deltaEl: React.ReactNode = null;
  if (delta != null && deltaFormat) {
    const eps = 1e-9;
    const dir = delta > eps ? 1 : delta < -eps ? -1 : 0;
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
    deltaEl = (
      <span className={`${color} text-[10px] mono ml-1`}>
        {sign}
        {deltaFormat(delta)}
      </span>
    );
  }

  let sparkColorClass = "text-muted";
  if (series.length >= 2) {
    const dir = series[series.length - 1].value - series[0].value;
    if (Math.abs(dir) > 1e-9) {
      sparkColorClass = higherIsBetter
        ? dir > 0
          ? "text-good"
          : "text-bad"
        : dir > 0
          ? "text-bad"
          : "text-good";
    }
  }

  return (
    <div className="flex items-center gap-2">
      <div className="flex-1 min-w-0">
        <div className="text-[9px] uppercase tracking-wider text-muted">
          {label}
        </div>
        <div className="mono text-[13px] text-text">
          {value}
          {deltaEl}
        </div>
      </div>
      <div className={sparkColorClass}>
        <Sparkline
          points={series}
          width={48}
          height={18}
          higherIsBetter={higherIsBetter}
          fill={false}
        />
      </div>
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

  const paramsStr =
    exp.params != null
      ? exp.params >= 1_000_000
        ? `${(exp.params / 1_000_000).toFixed(1)}M`
        : `${Math.round(exp.params / 1000)}K`
      : null;

  return (
    <section className="bg-panel border border-border rounded-xl overflow-hidden mb-3">
      {/* Header */}
      <div className="px-4 pt-3.5 pb-2 flex items-start gap-2">
        <div className="flex-1 min-w-0">
          <div className="flex items-center gap-2 flex-wrap">
            <h2 className="text-[15px] font-semibold text-text truncate">
              {exp.name}
            </h2>
            {exp.finalized && (
              <span className="text-[10px] text-accent border border-accent/40 rounded px-1.5 py-0.5 uppercase tracking-wide font-semibold">
                done g{exp.concluded_gen}
              </span>
            )}
          </div>
          <div className="text-[11px] text-muted mono mt-0.5 truncate">
            {exp.method}
            {paramsStr ? ` · ${paramsStr}` : ""}
            {exp.parent ? ` · ← ${exp.parent}` : ""}
            {exp.encounter_set ? ` · ${exp.encounter_set}` : ""}
          </div>
        </div>
        <StatusPill
          state={status.state}
          label={phase && status.state === "RUNNING" ? phase : undefined}
        />
      </div>

      {/* Progress strip */}
      <div className="px-4 pb-3">
        <div className="flex items-baseline justify-between text-[11px] mono mb-1">
          <span className="text-text">
            <span className="font-semibold">gen {gen}</span>
            <span className="text-muted">/{total}</span>
            {pct > 0 && (
              <span className="text-muted"> · {pct}%</span>
            )}
          </span>
          <span className="text-muted">
            {status.age_s != null
              ? `${formatDuration(status.age_s)} ago`
              : ""}
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

      {/* Primary metrics: 2×2 grid with sparklines */}
      <div className="px-3 pb-3 grid grid-cols-2 gap-2">
        <MetricCell
          label="P-Eval"
          value={
            ev?.passed != null && ev?.total ? (
              <>
                {ev.passed}
                <span className="text-muted text-sm">/{ev.total}</span>
              </>
            ) : (
              "—"
            )
          }
          hint={
            exp.peak_eval?.passed != null && exp.peak_eval?.total
              ? `peak ${exp.peak_eval.passed}/${exp.peak_eval.total}${
                  exp.peak_eval.gen != null
                    ? ` · g${exp.peak_eval.gen}`
                    : ""
                }`
              : undefined
          }
          delta={exp.eval_delta}
          deltaFormat={(d) => `${(d * 100).toFixed(1)}%`}
          series={exp.eval_series}
          peakGen={exp.peak_eval?.gen ?? null}
        />
        <MetricCell
          label="V-Eval"
          value={
            vev?.passed != null && vev?.total ? (
              <>
                {vev.passed}
                <span className="text-muted text-sm">/{vev.total}</span>
              </>
            ) : (
              "—"
            )
          }
          hint={
            exp.peak_value_eval?.passed != null &&
            exp.peak_value_eval?.total
              ? `peak ${exp.peak_value_eval.passed}/${exp.peak_value_eval.total}${
                  exp.peak_value_eval.gen != null
                    ? ` · g${exp.peak_value_eval.gen}`
                    : ""
                }`
              : undefined
          }
          delta={exp.value_eval_delta}
          deltaFormat={(d) => `${(d * 100).toFixed(1)}%`}
          series={exp.value_eval_series}
          peakGen={exp.peak_value_eval?.gen ?? null}
        />
        <MetricCell
          label="WR (last 10)"
          value={
            exp.win_rate_last10 != null
              ? formatPct(exp.win_rate_last10, 1)
              : "—"
          }
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
          series={exp.wr_series}
          peakGen={exp.best_win_rate_gen}
        />
        <MetricCell
          label="Rescue"
          value={
            mev?.rescue_rate != null
              ? `${mev.rescue_rate >= 0 ? "+" : ""}${(mev.rescue_rate * 100).toFixed(0)}%`
              : "—"
          }
          hint={
            mev?.clean != null
              ? `CL${mev.clean} · EC${mev.echo ?? 0} · FX${mev.fixed ?? 0}`
              : undefined
          }
          delta={exp.rescue_delta}
          deltaFormat={(d) => `${(d * 100).toFixed(0)}%`}
          series={exp.rescue_series}
        />
      </div>

      {/* Secondary: losses */}
      <div className="px-4 py-2.5 border-t border-border grid grid-cols-2 gap-3">
        <MiniStat
          label="P-loss"
          value={formatNum(exp.policy_loss_last10, 3)}
          delta={exp.policy_loss_delta}
          deltaFormat={(d) => d.toFixed(3)}
          higherIsBetter={false}
          series={exp.p_loss_series}
        />
        <MiniStat
          label="V-loss"
          value={formatNum(exp.value_loss_last10, 3)}
          delta={exp.value_loss_delta}
          deltaFormat={(d) => d.toFixed(3)}
          higherIsBetter={false}
          series={exp.v_loss_series}
        />
      </div>

      {/* Echo-chamber telemetry (shown only when present) */}
      {(exp.kl_mcts_net_last10 != null ||
        exp.top1_agree_last10 != null ||
        exp.value_corr_last10 != null) && (
        <div className="px-4 py-2.5 border-t border-border grid grid-cols-3 gap-3">
          <MiniStat
            label="KL(mcts‖net)"
            value={formatNum(exp.kl_mcts_net_last10, 3)}
            delta={exp.kl_mcts_net_delta}
            deltaFormat={(d) => d.toFixed(3)}
            higherIsBetter={false}
            series={exp.kl_series}
          />
          <MiniStat
            label="top1-agree"
            value={formatPct(exp.top1_agree_last10, 0)}
            delta={exp.top1_agree_delta}
            deltaFormat={(d) => `${(d * 100).toFixed(1)}%`}
            higherIsBetter={false}
            series={exp.top1_series}
          />
          <MiniStat
            label="V corr"
            value={formatNum(exp.value_corr_last10, 2)}
            delta={exp.value_corr_delta}
            deltaFormat={(d) => d.toFixed(3)}
            series={exp.vcorr_series}
          />
        </div>
      )}
    </section>
  );
}
