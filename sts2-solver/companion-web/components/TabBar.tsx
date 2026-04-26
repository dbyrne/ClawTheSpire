"use client";

import Link from "next/link";
import { usePathname } from "next/navigation";

const TABS = [
  { href: "/", label: "Live", icon: "◆" },
  { href: "/distill", label: "Distill", icon: "⌬" },
  { href: "/benchmarks", label: "Benchmarks", icon: "▤" },
  { href: "/leaderboard", label: "Evals", icon: "★" },
  { href: "/label", label: "Label", icon: "✎" },
];

export default function TabBar() {
  const pathname = usePathname();
  return (
    <nav
      className="fixed bottom-0 left-0 right-0 border-t border-border bg-panel/95 backdrop-blur"
      style={{ paddingBottom: "env(safe-area-inset-bottom, 0px)" }}
    >
      <div className="max-w-3xl mx-auto flex">
        {TABS.map((t) => {
          const active =
            t.href === "/"
              ? pathname === "/"
              : pathname.startsWith(t.href);
          return (
            <Link
              key={t.href}
              href={t.href}
              className={`flex-1 flex flex-col items-center py-3 text-xs font-medium transition-colors ${
                active
                  ? "text-accent"
                  : "text-muted hover:text-text"
              }`}
            >
              <span className="text-lg leading-none mb-1">{t.icon}</span>
              <span>{t.label}</span>
            </Link>
          );
        })}
      </div>
    </nav>
  );
}
