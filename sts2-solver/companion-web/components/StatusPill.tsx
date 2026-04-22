import { LiveState } from "../lib/types";

const STATE_CLASS: Record<LiveState, string> = {
  RUNNING: "bg-good/20 text-good border-good/40",
  STALLED: "bg-warn/20 text-warn border-warn/40",
  STOPPED: "bg-bad/20 text-bad border-bad/40",
  UNKNOWN: "bg-muted/20 text-muted border-muted/40",
};

export default function StatusPill({
  state,
  label,
}: {
  state: LiveState;
  label?: string;
}) {
  return (
    <span
      className={`inline-flex items-center gap-1.5 px-2 py-0.5 rounded-full border text-[11px] font-semibold uppercase tracking-wide mono ${STATE_CLASS[state]}`}
    >
      <span className="inline-block w-1.5 h-1.5 rounded-full bg-current" />
      {label ?? state}
    </span>
  );
}
