import type { Config } from "tailwindcss";

const config: Config = {
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        bg: "#0b0e13",
        panel: "#151a22",
        panel2: "#1c222c",
        border: "#2a313d",
        text: "#e6e8ed",
        muted: "#8a93a6",
        accent: "#4f8cff",
        good: "#3fb950",
        warn: "#d29922",
        bad: "#f85149",
        info: "#58a6ff",
      },
    },
  },
  plugins: [],
};

export default config;
