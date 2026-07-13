/** @type {import('tailwindcss').Config} */
export default {
  content: ["./index.html", "./src/**/*.{ts,tsx}"],
  // Preflight would reset the existing dark-panel styles in styles.css. We
  // keep our hand-rolled base and let Tailwind add utilities/components on
  // top — future pages can opt into utility classes freely.
  corePlugins: {
    preflight: false,
  },
  theme: {
    extend: {
      // Bridge our CSS variables so utility classes reuse the same tokens
      // (e.g. `bg-panel` / `text-muted`) instead of drifting to Tailwind's
      // default palette.
      colors: {
        bg: "var(--bg)",
        panel: "var(--panel)",
        "panel-2": "var(--panel-2)",
        border: "var(--border)",
        text: "var(--text)",
        muted: "var(--muted)",
        accent: "var(--accent)",
        "accent-dim": "var(--accent-dim)",
        ok: "var(--ok)",
        warn: "var(--warn)",
        err: "var(--err)",
      },
      fontFamily: {
        sans: "var(--sans)",
        mono: "var(--mono)",
      },
      borderRadius: {
        panel: "var(--radius)",
      },
    },
  },
  plugins: [],
};
