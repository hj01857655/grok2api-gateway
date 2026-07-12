import { useCallback, useEffect, useState } from "react";
import { api } from "../api";

type ModelRow = {
  id?: string;
  object?: string;
  owned_by?: string;
  root?: string;
};

export function ModelsPage() {
  const [items, setItems] = useState<ModelRow[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);

  const load = useCallback(async () => {
    setLoading(true);
    setError(null);
    try {
      const data = await api<{ ok?: boolean; models?: { data?: ModelRow[] } }>(
        "/admin/api/models",
      );
      const list = data.models?.data || [];
      setItems(Array.isArray(list) ? list : []);
    } catch (e) {
      setError(e instanceof Error ? e.message : String(e));
      setItems([]);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
  }, [load]);

  return (
    <>
      <h1 className="page-title">模型</h1>
      <p className="page-sub">经网关聚合的上游 /v1/models（admin 门禁）</p>

      <div className="row" style={{ marginTop: 0, marginBottom: 14 }}>
        <button type="button" className="primary" onClick={() => void load()} disabled={loading}>
          {loading ? "加载中…" : "刷新"}
        </button>
      </div>

      {error && <div className="msg err">{error}</div>}

      <div className="card">
        <h2>列表 ({items.length})</h2>
        {items.length === 0 ? (
          <p className="empty">{loading ? "加载中…" : "暂无模型 — 先在「账号」配置官方 Grok 凭据"}</p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>id</th>
                  <th>owned_by</th>
                  <th>root / alias</th>
                </tr>
              </thead>
              <tbody>
                {items.map((m) => (
                  <tr key={m.id || Math.random()}>
                    <td style={{ fontFamily: "var(--mono)" }}>{m.id || "—"}</td>
                    <td>{m.owned_by || "—"}</td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: "0.8rem" }}>
                      {m.root || "—"}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </>
  );
}
