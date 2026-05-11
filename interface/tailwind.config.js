/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    borderRadius: {
      none: "0",
      sm: "0.125rem",
      DEFAULT: "0.25rem",
      md: "0.375rem",
      lg: "0.5rem",
      xl: "0.5rem",
      "2xl": "0.5rem",
      full: "9999px",
    },
    extend: {
      colors: {
        maroon: "#610000",
        gold: "#735c00",
        parchment: "#fbfbe2",
        ink: "#1b1d0e",
        "paper-dim": "#efefd7",
      },
      fontFamily: {
        hamlet: ['"Newsreader"', "serif"],
        body: ['"Noto Serif"', "serif"],
      },
    },
  },
  plugins: [],
};
