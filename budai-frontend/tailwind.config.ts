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
        app: {
          bg: "#101115",
          surface: "#1A1C24",
          sidebar: "#14151B",
          border: "#2A2D35",
        },
        brand: {
          blue: "#3D73FF",
          orange: "#FF8A4C",
          pink: "#FF5E98",
          green: "#00E096",
        },
        text: {
          main: "#FFFFFF",
          muted: "#8B8E98",
        },
      },
    },
  },
  plugins: [],
};

export default config;
