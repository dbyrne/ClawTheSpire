import type { Metadata, Viewport } from "next";
import TabBar from "../components/TabBar";
import "./globals.css";

export const metadata: Metadata = {
  title: "BetaOne Companion",
  description: "Monitor STS2 experiment training, benchmarks, and evals.",
};

export const viewport: Viewport = {
  width: "device-width",
  initialScale: 1,
  maximumScale: 1,
  userScalable: false,
  themeColor: "#0b0e13",
};

export default function RootLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return (
    <html lang="en">
      <body>
        <main className="max-w-3xl mx-auto px-4 pt-4">{children}</main>
        <TabBar />
      </body>
    </html>
  );
}
