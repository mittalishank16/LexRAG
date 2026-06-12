// tailwind.config.ts
import type { Config } from "tailwindcss";

const config: Config = {
  content: [
    "./app/**/*.{ts,tsx}",
    "./components/**/*.{ts,tsx}",
  ],
  theme: {
    extend: {
      colors: {
        brand: {
          dark:   "#0A0F1E",
          navy:   "#1F3864",
          blue:   "#2E75B6",
          light:  "#7BB3E0",
        },
      },
      fontFamily: {
        serif: ["Georgia", "Cambria", "Times New Roman", "serif"],
        mono:  ["Courier New", "Courier", "monospace"],
      },
    },
  },
  plugins: [],
};

export default config;
