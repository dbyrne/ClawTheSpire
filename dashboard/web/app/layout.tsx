import type { Metadata } from "next";
import { Orbitron, Rajdhani, JetBrains_Mono } from "next/font/google";
import "./globals.css";

const orbitron = Orbitron({
  variable: "--font-orbitron",
  subsets: ["latin"],
  weight: ["400", "500", "700", "800", "900"],
});

const rajdhani = Rajdhani({
  variable: "--font-rajdhani",
  subsets: ["latin"],
  weight: ["400", "500", "600", "700"],
});

const jetbrains = JetBrains_Mono({
  variable: "--font-jetbrains",
  subsets: ["latin"],
  weight: ["400", "500", "700"],
});

export const metadata: Metadata = {
  title: "STS2 Oracle",
  description: "Slay the Spire 2 AI — Neural Network Run Tracker",
};

export default function RootLayout({
  children,
}: Readonly<{ children: React.ReactNode }>) {
  return (
    <html lang="en" className={`${orbitron.variable} ${rajdhani.variable} ${jetbrains.variable} h-full`}>
      <body className="min-h-full flex flex-col relative">
        {/* ── Main content ── */}
        <main className="flex-1 relative z-10 w-full max-w-[1920px] mx-auto px-3 py-2">
          {children}
        </main>
      </body>
    </html>
  );
}
