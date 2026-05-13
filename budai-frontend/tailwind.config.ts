import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./pages/**/*.{js,ts,jsx,tsx,mdx}",
    "./components/**/*.{js,ts,jsx,tsx,mdx}",
    "./app/**/*.{js,ts,jsx,tsx,mdx}",
  ],
  theme: {
    extend: {
      colors: {
        obsidian: "#0d1516",
        "neon-cyan": "#00e5ff",
        "deep-pink": "#ff3366",
        app: {
          bg: "#0d1516",
          surface: "#111b1d",
          sidebar: "#0a1112",
          border: "rgba(255, 255, 255, 0.08)",
        },
        brand: {
          cyan: "#00e5ff",
          pink: "#ff3366",
          blue: "#3D73FF",
          orange: "#FF8A4C",
          green: "#00E096",
        },
        text: {
          main: "#FFFFFF",
          muted: "#8B8E98",
        },
      },
      fontFamily: {
        sans: ["Geist", "Inter", "sans-serif"],
        geist: ["Geist", "sans-serif"],
      },
    },
  },
  plugins: [],
};

export default config;
