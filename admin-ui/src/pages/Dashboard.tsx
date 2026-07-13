import { useCallback, useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api, type LogSummary, type Status } from "../api";

export function DashboardPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [summary, setSummary] = useState<LogSummary | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const [st, sum] = await Promise.all([
        api<Status>("/admin/api/status"),
        api<LogSummary>("/admin/api/logs/summary").catch(() => null),
      ]);
      setStatus(st);
      setSummary(sum);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  const mode = status?.effective_upstream_mode || status?.upstream_mode || "—";
  const hasUpstream = !!status?.oauth_current || !!status?.upstream_key_configured;

  return (
    <>
      <h1 className="page-title">总览</h1>
      <p className="page-sub">网关状态与近期请求摘要</p>

      <div className="row !mt-0 mb-3.5">
        <button type="button" className="primary" onClick={() => void load()} disabled={loading}>
          {loading ? "加载中…" : "刷新"}
        </button>
        <span className={`badge ${hasUpstream ? "ok" : "warn"}`}>upstream: {mode}</span>
      </div>

      {error && <div className="msg err">{error}</div>}

      <div className="card">
        <h2>网关</h2>
        {status ? (
          <div className="kv">
            <div className="k">version</div>
            <div className="v">{status.version ?? "—"}</div>
            <div className="k">upstream_mode</div>
            <div className="v">
              {status.upstream_mode}
              {status.effective_upstream_mode &&
              status.effective_upstream_mode !== status.upstream_mode
                ? ` → ${status.effective_upstream_mode}`
                : ""}
            </div>
            <div className="k">oauth</div>
            <div className="v">
              {status.oauth_current
                ? status.oauth_current.email || status.oauth_current.sub || "ok"
                : "未配置"}
            </div>
            <div className="k">auths_dir</div>
            <div className="v">{status.oauth_auths_dir || "—"}</div>
          </div>
        ) : (
          !error && <p className="empty">{loading ? "加载中…" : "无数据"}</p>
        )}
        <div className="row">
          <Link className="btn" to="/accounts">
            管理账号
          </Link>
        </div>
      </div>

      <div className="card">
        <h2>请求日志摘要</h2>
        {summary ? (
          <div className="stats">
            <div className="stat">
              <div className="label">近 1 小时</div>
              <div className="value">{summary.last_1h?.count ?? 0}</div>
              <div className="label">
                4xx {summary.last_1h?.["4xx"] ?? 0} · 5xx {summary.last_1h?.["5xx"] ?? 0}
              </div>
            </div>
            <div className="stat">
              <div className="label">近 24 小时</div>
              <div className="value">{summary.last_24h?.count ?? 0}</div>
              <div className="label">
                4xx {summary.last_24h?.["4xx"] ?? 0} · 5xx {summary.last_24h?.["5xx"] ?? 0}
              </div>
            </div>
          </div>
        ) : (
          <p className="empty">暂无摘要（未启用日志或尚无请求）</p>
        )}
        <div className="row">
          <Link className="btn" to="/logs">
            查看日志
          </Link>
        </div>
      </div>
    </>
  );
}
