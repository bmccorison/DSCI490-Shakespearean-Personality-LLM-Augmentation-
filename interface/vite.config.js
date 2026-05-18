import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";

const backendPort = process.env.BACKEND_PORT || "8000";

export default defineConfig({
  plugins: [react()],
  server: {
    host: "localhost",
    port: 6969,
    proxy: {
      "/api": {
        target: `http://127.0.0.1:${backendPort}`,
        changeOrigin: true,
      },
    },
  },
});
