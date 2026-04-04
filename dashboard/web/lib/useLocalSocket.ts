"use client";

import { useEffect, useState, useRef, useCallback } from "react";
import type { Run, RunEvent } from "./types";

const API_BASE = "http://localhost:8765";
const WS_URL = "ws://localhost:8765/ws";

interface LocalSocketState {
  runs: Run[];
  liveRun: Run | null;
  liveEvents: RunEvent[];
  connected: boolean;
  loading: boolean;
}

/**
 * Connects to the local EventServer for real-time run data.
 * Fetches historical runs + events via HTTP, streams live updates via WS.
 */
export function useLocalSocket(): LocalSocketState {
  const [runs, setRuns] = useState<Run[]>([]);
  const [wsRun, setWsRun] = useState<Run | null>(null);
  const [wsEvents, setWsEvents] = useState<RunEvent[] | null>(null);
  const [httpEvents, setHttpEvents] = useState<RunEvent[]>([]);
  const [connected, setConnected] = useState(false);
  const [loading, setLoading] = useState(true);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelay = useRef(1000);
  const unmounted = useRef(false);
  const heroRunIdRef = useRef<string | null>(null);

  // Derive hero run: WS active run takes priority, otherwise latest from HTTP
  const heroRun = wsRun ?? (runs.length > 0
    ? (runs.find((r) => !r.outcome) ?? runs[runs.length - 1])
    : null);

  // Events: WS events take priority when available
  const liveEvents = wsEvents ?? httpEvents;

  // Fetch historical runs from HTTP API
  const fetchRuns = useCallback(() => {
    fetch(`${API_BASE}/api/runs?limit=500`)
      .then((r) => r.json())
      .then((data: Run[]) => {
        if (!unmounted.current && Array.isArray(data)) {
          setRuns(data);
          setLoading(false);
        }
      })
      .catch(() => setLoading(false));
  }, []);

  // Fetch events for a specific run from HTTP API
  const fetchEvents = useCallback((runId: string) => {
    fetch(`${API_BASE}/api/runs/${runId}/events`)
      .then((r) => r.json())
      .then((data: RunEvent[]) => {
        if (!unmounted.current && Array.isArray(data)) {
          setHttpEvents(data);
        }
      })
      .catch(() => {});
  }, []);

  // When hero run changes, fetch its events via HTTP
  useEffect(() => {
    const id = heroRun?.run_id ?? null;
    if (id && id !== heroRunIdRef.current) {
      heroRunIdRef.current = id;
      fetchEvents(id);
    }
  }, [heroRun?.run_id, fetchEvents]);

  // Connect WebSocket
  const connect = useCallback(() => {
    if (unmounted.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (unmounted.current) { ws.close(); return; }
      setConnected(true);
      reconnectDelay.current = 1000;
    };

    ws.onmessage = (e) => {
      if (unmounted.current) return;
      try {
        const msg = JSON.parse(e.data);
        switch (msg.type) {
          case "snapshot":
            if (msg.run) {
              setWsRun(msg.run);
              setRuns((prev) => {
                const idx = prev.findIndex((r) => r.run_id === msg.run.run_id);
                if (idx >= 0) {
                  const next = [...prev];
                  next[idx] = msg.run;
                  return next;
                }
                return [...prev, msg.run];
              });
            }
            if (msg.events?.length) {
              setWsEvents(msg.events);
            }
            break;

          case "run_start":
            setWsRun(msg.data);
            setWsEvents([]);
            setRuns((prev) => [...prev, msg.data]);
            break;

          case "run_update":
            setWsRun(msg.data);
            setRuns((prev) => {
              const idx = prev.findIndex((r) => r.run_id === msg.data.run_id);
              if (idx >= 0) {
                const next = [...prev];
                next[idx] = msg.data;
                return next;
              }
              return [...prev, msg.data];
            });
            break;

          case "event":
            // If WS events are active, append there; otherwise start from HTTP events
            setWsEvents((prev) => prev != null ? [...prev, msg.data] : null);
            // Also append to HTTP events as fallback
            setHttpEvents((prev) => [...prev, msg.data]);
            break;
        }
      } catch {
        // ignore malformed messages
      }
    };

    ws.onclose = () => {
      if (unmounted.current) return;
      setConnected(false);
      const delay = reconnectDelay.current;
      reconnectDelay.current = Math.min(delay * 2, 30000);
      setTimeout(connect, delay);
    };

    ws.onerror = () => {};
  }, []);

  useEffect(() => {
    unmounted.current = false;
    fetchRuns();
    connect();
    // Refresh full runs list every 10s to catch completed runs
    const interval = setInterval(fetchRuns, 10_000);
    return () => {
      unmounted.current = true;
      clearInterval(interval);
      wsRef.current?.close();
    };
  }, [fetchRuns, connect]);

  return { runs, liveRun: heroRun, liveEvents, connected, loading };
}

/**
 * Fetch a specific run and its events from the local HTTP API.
 * Subscribes to WS for live updates if the run is in progress.
 */
export function useRunDetail(runId: string) {
  const [run, setRun] = useState<Run | null>(null);
  const [events, setEvents] = useState<RunEvent[]>([]);
  const [loading, setLoading] = useState(true);
  const [connected, setConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const reconnectDelay = useRef(1000);
  const unmounted = useRef(false);

  const fetchData = useCallback(() => {
    Promise.all([
      fetch(`${API_BASE}/api/runs/${runId}`).then((r) => r.ok ? r.json() : null),
      fetch(`${API_BASE}/api/runs/${runId}/events`).then((r) => r.ok ? r.json() : []),
    ])
      .then(([runData, eventsData]) => {
        if (!unmounted.current) {
          setRun(runData);
          if (Array.isArray(eventsData)) setEvents(eventsData);
          setLoading(false);
        }
      })
      .catch(() => setLoading(false));
  }, [runId]);

  const connect = useCallback(() => {
    if (unmounted.current) return;
    if (wsRef.current?.readyState === WebSocket.OPEN) return;

    const ws = new WebSocket(WS_URL);
    wsRef.current = ws;

    ws.onopen = () => {
      if (unmounted.current) { ws.close(); return; }
      setConnected(true);
      reconnectDelay.current = 1000;
    };

    ws.onmessage = (e) => {
      if (unmounted.current) return;
      try {
        const msg = JSON.parse(e.data);
        switch (msg.type) {
          case "event":
            if (msg.data.run_id === runId) {
              setEvents((prev) => [...prev, msg.data]);
            }
            break;
          case "run_update":
            if (msg.data.run_id === runId) {
              setRun(msg.data);
            }
            break;
          case "snapshot":
            if (msg.run?.run_id === runId) {
              setRun(msg.run);
              if (msg.events?.length) setEvents(msg.events);
            }
            break;
        }
      } catch {
        // ignore
      }
    };

    ws.onclose = () => {
      if (unmounted.current) return;
      setConnected(false);
      const delay = reconnectDelay.current;
      reconnectDelay.current = Math.min(delay * 2, 30000);
      setTimeout(connect, delay);
    };

    ws.onerror = () => {};
  }, [runId]);

  useEffect(() => {
    unmounted.current = false;
    fetchData();
    connect();
    return () => {
      unmounted.current = true;
      wsRef.current?.close();
    };
  }, [fetchData, connect]);

  return { run, events, loading, connected };
}
