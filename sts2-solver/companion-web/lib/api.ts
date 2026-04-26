export const fetcher = async (url: string) => {
  const res = await fetch(url);
  if (!res.ok) {
    throw new Error(`${res.status} ${res.statusText}`);
  }
  return res.json();
};

export function formatDuration(secs: number | null | undefined): string {
  if (secs == null) return "-";
  if (secs < 60) return `${Math.round(secs)}s`;
  if (secs < 3600) return `${Math.round(secs / 60)}m`;
  if (secs < 86400) return `${(secs / 3600).toFixed(1)}h`;
  return `${Math.round(secs / 86400)}d`;
}

export function formatPct(x: number | null | undefined, digits = 1): string {
  if (x == null) return "-";
  return `${(x * 100).toFixed(digits)}%`;
}

export function formatNum(
  x: number | null | undefined,
  digits = 3,
): string {
  if (x == null) return "-";
  return x.toFixed(digits);
}

export function formatCost(x: number | null | undefined): string {
  if (x == null) return "-";
  if (x < 10) return `$${x.toFixed(2)}`;
  if (x < 100) return `$${x.toFixed(1)}`;
  return `$${Math.round(x)}`;
}
