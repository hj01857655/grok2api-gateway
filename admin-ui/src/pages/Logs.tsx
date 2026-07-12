import { useCallback, useEffect, useState } from "react";
import { api, type LogItem } from "../api";

export function LogsPage() {
  const [items, setItems] = useState<LogItem[]>([]);
  const [total, setTotal] = useState(0);
  const [offset, setOffset] = useState(0);
  const [pathPrefix, setPathPrefix] = useState("");
  const [statusMin, setStatusMin] = useState("");
  const [statusMax, setStatusMax] = useState("");
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const limit = 50;

  const load = useCallback(
    async (nextOffset = offset) => {
      setLoading(true);
      setError(null);
      const q = new URLSearchParams();
      q.set("limit", String(limit));
      q.set("offset", String(nextOffset));
      if (pathPrefix.trim()) q.set("path_prefix", pathPrefix.trim());
      if (statusMin.trim()) q.set("status_min", statusMin.trim());
      if (statusMax.trim()) q.set("status_max", statusMax.trim());
      try {
        const data = await api<{ items?: LogItem[]; total?: number }>(
          `/admin/api/logs?${q.toString()}`,
        );
        setItems(data.items || []);
        setTotal(data.total ?? 0);
        setOffset(nextOffset);
      } catch (e) {
        setError(e instanceof Error ? e.message : String(e));
        setItems([]);
      } finally {
        setLoading(false);
      }
    },
    [offset, pathPrefix, statusMin, statusMax],
  );

  useEffect(() => {
    void load(0);
    // initial load only
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, []);

  const page = Math.floor(offset / limit) + 1;
  const pages = Math.max(1, Math.ceil(total / limit));

  return (
    <>
      <h1 className="page-title">请求日志</h1>
      <p className="page-sub">元数据 JSONL（默认不含 body）· 按日/大小轮转</p>

      <div className="card">
        <h2>筛选</h2>
        <div className="grid2">
          <div className="field">
            <label>path 前缀</label>
            <input
              value={pathPrefix}
              onChange={(e) => setPathPrefix(e.target.value)}
              placeholder="/v1/"
            />
          </div>
          <div className="field">
            <label>status 范围</label>
            <div className="row" style={{ marginTop: 0 }}>
              <input
                style={{ maxWidth: 100 }}
                value={statusMin}
                onChange={(e) => setStatusMin(e.target.value)}
                placeholder="min"
              />
              <span className="empty">–</span>
              <input
                style={{ maxWidth: 100 }}
                value={statusMax}
                onChange={(e) => setStatusMax(e.target.value)}
                placeholder="max"
              />
            </div>
          </div>
        </div>
        <div className="row">
          <button type="button" className="primary" onClick={() => void load(0)} disabled={loading}>
            {loading ? "加载中…" : "查询"}
          </button>
          <button
            type="button"
            disabled={loading || offset <= 0}
            onClick={() => void load(Math.max(0, offset - limit))}
          >
            上一页
          </button>
          <button
            type="button"
            disabled={loading || offset + limit >= total}
            onClick={() => void load(offset + limit)}
          >
            下一页
          </button>
          <span className="empty">
            共 {total} · 第 {page}/{pages} 页
          </span>
        </div>
        {error && <div className="msg err">{error}</div>}
      </div>

      <div className="card">
        <h2>记录</h2>
        {items.length === 0 ? (
          <p className="empty">{loading ? "加载中…" : "暂无记录"}</p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>时间</th>
                  <th>方法</th>
                  <th>路径</th>
                  <th>状态</th>
                  <th>耗时 ms</th>
                  <th>模型</th>
                  <th>错误</th>
                </tr>
              </thead>
              <tbody>
                {items.map((r, i) => {
                  const st = r.status ?? 0;
                  const badge =
                    st >= 500 ? "err" : st >= 400 ? "warn" : st >= 200 ? "ok" : "";
                  return (
                    <tr key={`${r.ts}-${r.path}-${i}`}>
                      <td style={{ fontFamily: "var(--mono)", fontSize: "0.78rem" }}>
                        {r.ts || "—"}
                      </td>
                      <td>{r.method || "—"}</td>
                      <td style={{ fontFamily: "var(--mono)", fontSize: "0.8rem" }}>
                        {r.path || "—"}
                      </td>
                      <td>
                        <span className={`badge ${badge}`}>{st || "—"}</span>
                      </td>
                      <td>{r.duration_ms ?? "—"}</td>
                      <td style={{ fontFamily: "var(--mono)", fontSize: "0.8rem" }}>
                        {r.model || "—"}
                      </td>
                      <td style={{ color: "var(--err)", fontSize: "0.8rem" }}>
                        {r.error || ""}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
