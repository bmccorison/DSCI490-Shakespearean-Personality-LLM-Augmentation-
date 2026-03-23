/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{js,ts,jsx,tsx}"],
  theme: {
    extend: {
      colors: {
        maroon: "#A52E30",
        gold: "#FDD492",
        parchment: "#FFFDF9",
      },
      fontFamily: {
        hamlet: ['"HamletOrNot"', '"IM FELL English SC"', "serif"],
        body: ['"Cormorant Garamond"', "serif"],
      },
    },
  },
  plugins: [],
};
