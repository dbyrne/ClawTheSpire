import { readFile } from "fs/promises";
import { join } from "path";

const BASE = join(process.cwd(), "..", "..");
const PROGRESS_PATH = join(BASE, "alphazero_progress.json");
const HISTORY_PATH = join(BASE, "alphazero_history.jsonl");

export const dynamic = "force-dynamic";

export async function GET(request: Request) {
  const url = new URL(request.url);
  const type = url.searchParams.get("type");

  if (type === "history") {
    try {
      const raw = await readFile(HISTORY_PATH, "utf-8");
      const lines = raw.trim().split("\n").filter(Boolean);
      const history = lines.map((line) => JSON.parse(line));
      return Response.json(history);
    } catch {
      return Response.json([]);
    }
  }

  try {
    const raw = await readFile(PROGRESS_PATH, "utf-8");
    // Replace NaN/Infinity with null so JSON.parse doesn't choke
    const sanitized = raw.replace(/\bNaN\b/g, "null").replace(/\bInfinity\b/g, "null");
    return Response.json(JSON.parse(sanitized));
  } catch {
    return Response.json(null, { status: 404 });
  }
}
