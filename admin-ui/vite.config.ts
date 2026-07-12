import { defineConfig } from "vite";
import react from "@vitejs/plugin-react";
import path from "node:path";

export default defineConfig({
  plugins: [react()],
  base: "/admin/",
  server: {
    port: 5173,
    proxy: {
      "/admin/api": {
        target: "http://127.0.0.1:8787",
        changeOrigin: true,
      },
    },
  },
  build: {
    outDir: path.resolve(__dirname, "../app/static/admin-dist"),
    emptyOutDir: true,
  },
});
