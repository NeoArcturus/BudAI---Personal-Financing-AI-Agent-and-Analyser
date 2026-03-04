/** @type {import('tailwindcss').Config} */
module.exports = {
  darkMode: ["class"],
  content: ["./app/**/*.{ts,tsx}", "./components/**/*.{ts,tsx}"],
  theme: {
    extend: {
      colors: {
        background: "#0D1117", // Deep GitHub Dark
        foreground: "#F0F6FC",
        primary: "#00FFAA", // BudAI Neon Green
        secondary: "#161B22", // Lighter Card Background
      },
    },
  },
  plugins: [],
};
