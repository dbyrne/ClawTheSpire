import { SeriesPoint } from "../lib/types";

export default function Sparkline({
  points,
  peakGen,
  color = "currentColor",
  width = 88,
  height = 28,
  fill = true,
  higherIsBetter = true,
}: {
  points: SeriesPoint[];
  peakGen?: number | null;
  color?: string;
  width?: number;
  height?: number;
  fill?: boolean;
  higherIsBetter?: boolean;
}) {
  if (!points || points.length < 2) {
    return (
      <div
        className="text-muted text-[10px] mono"
        style={{ width, height, lineHeight: `${height}px` }}
      >
        —
      </div>
    );
  }

  const vals = points.map((p) => p.value);
  const min = Math.min(...vals);
  const max = Math.max(...vals);
  const range = max - min || 1;
  const pad = 3;
  const innerH = height - pad * 2;
  const step = points.length === 1 ? 0 : width / (points.length - 1);
  const yFor = (v: number) => pad + innerH - ((v - min) / range) * innerH;

  const linePath = points
    .map(
      (p, i) =>
        `${i === 0 ? "M" : "L"}${(i * step).toFixed(2)},${yFor(p.value).toFixed(2)}`,
    )
    .join(" ");
  const areaPath = `${linePath} L${width},${height} L0,${height} Z`;

  // Peak: argmax under higherIsBetter, else argmin. Then match gen if provided.
  let peakIdx = higherIsBetter
    ? vals.indexOf(max)
    : vals.indexOf(min);
  if (peakGen != null) {
    const idxByGen = points.findIndex((p) => p.gen === peakGen);
    if (idxByGen >= 0) peakIdx = idxByGen;
  }

  const lastIdx = points.length - 1;
  const lastX = lastIdx * step;
  const lastY = yFor(points[lastIdx].value);
  const peakX = peakIdx * step;
  const peakY = yFor(points[peakIdx].value);

  return (
    <svg
      width={width}
      height={height}
      viewBox={`0 0 ${width} ${height}`}
      className="inline-block align-middle"
      preserveAspectRatio="none"
    >
      {fill && (
        <path d={areaPath} fill={color} fillOpacity={0.12} stroke="none" />
      )}
      <path
        d={linePath}
        fill="none"
        stroke={color}
        strokeWidth={1.4}
        strokeLinejoin="round"
        strokeLinecap="round"
        vectorEffect="non-scaling-stroke"
      />
      {peakIdx !== lastIdx && (
        <circle
          cx={peakX}
          cy={peakY}
          r={2.2}
          fill="transparent"
          stroke={color}
          strokeWidth={1.2}
          opacity={0.7}
        />
      )}
      <circle cx={lastX} cy={lastY} r={2.2} fill={color} />
    </svg>
  );
}
