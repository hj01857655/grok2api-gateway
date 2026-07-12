import { useState } from "react";
import { api, getAdminKey, setAdminKey, type Status } from "../api";

export function SettingsPage() {
  const [key, setKey] = useState(getAdminKey());
  const [msg, setMsg] = useState<{ text: string; kind: "ok" | "err" | "info" } | null>(null);
  const [reloading, setReloading] = useState(false);

  function saveKey() {
    setAdminKey(key);
    setMsg({ text: "门禁密钥已写入 localStorage", kind: "ok" });
  }

  async function reloadSettings() {
    setReloading(true);
    setMsg(null);
    try {
      const data = await api<{ ok?: boolean; status?: Status }>("/admin/api/reload-settings", {
        method: "POST",
      });
      setMsg({
        text: `设置已重载 · version ${data.status?.version ?? "—"} · mode ${data.status?.effective_upstream_mode || data.status?.upstream_mode || "—"}`,
        kind: "ok",
      });
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : String(e), kind: "err" });
    } finally {
      setReloading(false);
    }
  }

  return (
    <>
      <h1 className="page-title">设置</h1>
      <p className="page-sub">门禁密钥保存在浏览器 localStorage；进程配置仍在 .env</p>

      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}

      <div className="card">
        <h2>Admin 门禁</h2>
        <p className="hint">
          与 <code>GROK2API_API_KEY</code> 相同。请求头：Authorization Bearer / x-admin-key。
        </p>
        <div className="field">
          <label>门禁密钥</label>
          <input
            type="password"
            value={key}
            onChange={(e) => setKey(e.target.value)}
            placeholder="sk-…"
            autoComplete="off"
          />
        </div>
        <div className="row">
          <button type="button" className="primary" onClick={saveKey}>
            保存到本机
          </button>
        </div>
      </div>

      <div className="card">
        <h2>进程设置</h2>
        <p className="hint">
          HOST / PORT / UPSTREAM_MODE / 日志等写在 <code>.env</code>。改完后可触发服务端
          reload（清 Settings 缓存，不重启进程）。
        </p>
        <div className="row">
          <button
            type="button"
            className="primary"
            disabled={reloading}
            onClick={() => void reloadSettings()}
          >
            {reloading ? "重载中…" : "reload-settings"}
          </button>
        </div>
      </div>

      <div className="card">
        <h2>说明</h2>
        <ul className="hint" style={{ paddingLeft: 18, margin: 0 }}>
          <li>官方 Grok 凭据 → ~/.grok2api/auths/xai-*.json</li>
          <li>请求日志 → ~/.grok2api/logs/requests-*.jsonl</li>
          <li>
            开发 UI：<code>cd admin-ui &amp;&amp; npm run dev</code>（代理 /admin/api → :8787）
          </li>
          <li>
            生产构建：<code>npm run build</code> → app/static/admin-dist，由 FastAPI 挂 /admin
          </li>
        </ul>
      </div>
    </>
  );
}
