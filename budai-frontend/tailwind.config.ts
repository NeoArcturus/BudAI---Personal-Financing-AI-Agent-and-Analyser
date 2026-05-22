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
        obsidian: "var(--background)",
        "neon-cyan": "var(--primary)",
        "deep-pink": "#ff3366",
        app: {
          bg: "var(--background)",
          surface: "var(--card)",
          sidebar: "var(--sidebar)",
          border: "var(--border)",
        },
        brand: {
          cyan: "var(--primary)",
          pink: "#ff3366",
          blue: "#3D73FF",
          orange: "#FF8A4C",
          green: "#00E096",
        },
        text: {
          main: "var(--foreground)",
          muted: "var(--muted-foreground)",
        },
      },
      fontFamily: {
        sans: ["var(--font-jakarta)", "sans-serif"],
        geist: ["var(--font-jakarta)", "sans-serif"],
      },
      animation: {
        aurora: "aurora 60s cubic-bezier(0.4, 0.0, 0.2, 1) infinite",
      },
      keyframes: {
        aurora: {
          "0%, 100%": {
            backgroundPosition: "50% 50%, 50% 50%",
          },
          "20%": {
            backgroundPosition: "50% 50%, 120% 30%",
          },
          "40%": {
            backgroundPosition: "50% 50%, 250% 70%",
          },
          "60%": {
            backgroundPosition: "50% 50%, 80% 80%",
          },
          "80%": {
            backgroundPosition: "50% 50%, 320% 20%",
          },
        },
      },
    },
  },
  plugins: [addVariablesForColors],
};

// This plugin adds each Tailwind color as a global CSS variable, e.g. var(--gray-200).
function addVariablesForColors({ addBase, theme }: any) {
  const allColors = theme("colors");
  const newVars = Object.fromEntries(
    Object.entries(allColors).map(([key, val]) => [`--${key}`, val])
  );

  addBase({
    ":root": newVars,
  });
}

export default config;
