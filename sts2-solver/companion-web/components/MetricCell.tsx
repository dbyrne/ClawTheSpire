import { SeriesPoint } from "../lib/types";
import Sparkline from "./Sparkline";

export default function MetricCell({
  label,
  value,
  hint,
  delta,
  deltaFormat,
  higherIsBetter = true,
  series = [],
  peakGen,
  accentClass = "text-accent",
}: {
  label: string;
  value: React.ReactNode;
  hint?: React.ReactNode;
  delta?: number | null;
  deltaFormat?: (d: number) => string;
  higherIsBetter?: boolean;
  series?: SeriesPoint[];
  peakGen?: number | null;
  accentClass?: string;
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
    const arrow = dir > 0 ? "↑" : dir < 0 ? "↓" : "·";
    deltaEl = (
      <span className={`${color} text-[10px] mono font-semibold`}>
        {arrow} {sign}
        {deltaFormat(delta)}
      </span>
    );
  }

  // Sparkline color follows deltaFormat's semantics when we have one;
  // otherwise inherits from accentClass (default blue). currentColor lets
  // us drive SVG stroke from the parent's Tailwind text color.
  let sparkColorClass = accentClass;
  if (series.length >= 2) {
    const last = series[series.length - 1].value;
    const first = series[0].value;
    const dir = last - first;
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
    <div className="bg-panel2/60 border border-border/60 rounded-lg p-2.5 flex flex-col gap-1">
      <div className="flex items-center justify-between">
        <span className="text-[10px] uppercase tracking-wider text-muted font-medium">
          {label}
        </span>
        {deltaEl}
      </div>
      <div className="flex items-end justify-between gap-2 min-h-[32px]">
        <div className="mono text-lg font-semibold text-text leading-none">
          {value}
        </div>
        <div className={sparkColorClass}>
          <Sparkline
            points={series}
            peakGen={peakGen ?? null}
            higherIsBetter={higherIsBetter}
          />
        </div>
      </div>
      {hint && (
        <div className="text-[10px] text-muted mono truncate">{hint}</div>
      )}
    </div>
  );
}
