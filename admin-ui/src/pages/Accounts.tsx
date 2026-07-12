import { useCallback, useEffect, useRef, useState } from "react";
import { api, getAdminKey, type OAuthFile, type Status } from "../api";

export function AccountsPage() {
  const [status, setStatus] = useState<Status | null>(null);
  const [msg, setMsg] = useState<{ text: string; kind: "ok" | "err" | "info" } | null>(null);
  const [loading, setLoading] = useState(true);
  const [usingApi, setUsingApi] = useState(false);
  const [loginUsingApi, setLoginUsingApi] = useState(false);
  const [importUsingApi, setImportUsingApi] = useState(false);
  const [jsonPaste, setJsonPaste] = useState("");
  const [userCode, setUserCode] = useState<string | null>(null);
  const [openUrl, setOpenUrl] = useState<string | null>(null);
  const [starting, setStarting] = useState(false);
  const pollRef = useRef<number | null>(null);
  const fileRef = useRef<HTMLInputElement>(null);

  const load = useCallback(async () => {
    setLoading(true);
    try {
      const st = await api<Status>("/admin/api/status");
      setStatus(st);
      setUsingApi(!!st.oauth_current?.using_api);
      setMsg(null);
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : String(e), kind: "err" });
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    void load();
    return () => {
      if (pollRef.current) window.clearInterval(pollRef.current);
    };
  }, [load]);

  async function selectFile(name: string) {
    try {
      await api("/admin/api/select", {
        method: "POST",
        json: { name, using_api: usingApi },
      });
      setMsg({ text: `已设为当前: ${name}`, kind: "ok" });
      await load();
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : String(e), kind: "err" });
    }
  }

  async function saveUsingApi() {
    try {
      await api("/admin/api/using-api", {
        method: "POST",
        json: { using_api: usingApi },
      });
      setMsg({ text: "using_api 已更新", kind: "ok" });
      await load();
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : String(e), kind: "err" });
    }
  }

  async function pollSession(id: string) {
    try {
      const data = await api<{
        status?: string;
        email?: string;
        credential_path?: string;
        error?: string;
      }>(`/admin/api/oauth/device/${encodeURIComponent(id)}`);
      if (data.status === "pending") return;
      if (pollRef.current) {
        window.clearInterval(pollRef.current);
        pollRef.current = null;
      }
      if (data.status === "success") {
        setMsg({
          text: `Device Code 成功: ${data.email || ""} → ${data.credential_path || ""}`,
          kind: "ok",
        });
        setUserCode(null);
        await load();
      } else {
        setMsg({ text: `失败: ${data.error || data.status}`, kind: "err" });
      }
    } catch (e) {
      const m = e instanceof Error ? e.message : String(e);
      if (m.includes("not found")) {
        if (pollRef.current) {
          window.clearInterval(pollRef.current);
          pollRef.current = null;
        }
        setMsg({ text: m, kind: "err" });
      }
    }
  }

  async function startLogin() {
    setStarting(true);
    setUserCode(null);
    setOpenUrl(null);
    try {
      const data = await api<{
        session_id?: string;
        user_code?: string;
        open_url?: string;
      }>("/admin/api/oauth/device/start", {
        method: "POST",
        json: { using_api: loginUsingApi },
      });
      setUserCode(data.user_code || "——");
      if (data.open_url) {
        setOpenUrl(data.open_url);
        window.open(data.open_url, "_blank", "noopener");
      }
      setMsg({ text: `等待授权… session=${data.session_id}`, kind: "info" });
      if (pollRef.current) window.clearInterval(pollRef.current);
      if (data.session_id) {
        pollRef.current = window.setInterval(() => void pollSession(data.session_id!), 2000);
      }
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : String(e), kind: "err" });
    } finally {
      setStarting(false);
    }
  }

  async function importPaste() {
    try {
      const data = await api<{ email?: string; path?: string }>("/admin/api/import", {
        method: "POST",
        json: {
          json_text: jsonPaste,
          using_api: importUsingApi ? true : null,
        },
      });
      setMsg({ text: `导入成功: ${data.email || data.path}`, kind: "ok" });
      await load();
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : String(e), kind: "err" });
    }
  }

  async function importFile() {
    const f = fileRef.current?.files?.[0];
    if (!f) {
      setMsg({ text: "请选择 xai-*.json", kind: "err" });
      return;
    }
    const fd = new FormData();
    fd.append("file", f);
    if (importUsingApi) fd.append("using_api", "true");
    try {
      const key = getAdminKey();
      const headers: Record<string, string> = {};
      if (key) {
        headers.Authorization = `Bearer ${key}`;
        headers["x-admin-key"] = key;
      }
      const res = await fetch("/admin/api/import/file", { method: "POST", headers, body: fd });
      const data = await res.json();
      if (!res.ok) {
        throw new Error(
          typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail || data),
        );
      }
      setMsg({ text: `导入成功: ${data.email || data.path}`, kind: "ok" });
      await load();
    } catch (e) {
      setMsg({ text: e instanceof Error ? e.message : String(e), kind: "err" });
    }
  }

  const cur = status?.oauth_current;
  const files: OAuthFile[] = status?.oauth_files || [];

  return (
    <>
      <h1 className="page-title">官方 Grok 账号</h1>
      <p className="page-sub">
        Device Code / 导入 <code>xai-*.json</code> → <code>{status?.oauth_auths_dir || "…/auths"}</code>
      </p>

      {msg && <div className={`msg ${msg.kind}`}>{msg.text}</div>}

      <div className="card">
        <h2>当前账号</h2>
        {loading && !status ? (
          <p className="empty">加载中…</p>
        ) : cur ? (
          <>
            <div className="kv">
              <div className="k">email</div>
              <div className="v">{cur.email || cur.sub || "—"}</div>
              <div className="k">expired</div>
              <div className="v">{cur.expired || "—"}</div>
              <div className="k">chat_base</div>
              <div className="v">{cur.chat_base || "—"}</div>
              <div className="k">tokens</div>
              <div className="v">
                access={String(!!cur.has_access)} refresh={String(!!cur.has_refresh)}
              </div>
            </div>
            <div className="row">
              <label className="check">
                <input
                  type="checkbox"
                  checked={usingApi}
                  onChange={(e) => setUsingApi(e.target.checked)}
                />
                using_api
              </label>
              <button type="button" onClick={() => void saveUsingApi()}>
                保存 using_api
              </button>
              <button type="button" onClick={() => void load()}>
                刷新
              </button>
            </div>
          </>
        ) : (
          <p className="empty">暂无当前账号 — 下方登录或导入</p>
        )}
      </div>

      <div className="card">
        <h2>Device Code 登录</h2>
        <p className="hint">仅官方 xAI/Grok。浏览器打开授权页并输入用户码。</p>
        <label className="check">
          <input
            type="checkbox"
            checked={loginUsingApi}
            onChange={(e) => setLoginUsingApi(e.target.checked)}
          />
          using_api
        </label>
        <div className="row">
          <button
            type="button"
            className="primary"
            disabled={starting}
            onClick={() => void startLogin()}
          >
            {starting ? "启动中…" : "开始登录"}
          </button>
          {openUrl && (
            <a className="btn" href={openUrl} target="_blank" rel="noreferrer">
              打开授权页
            </a>
          )}
        </div>
        {userCode && <div className="code-big">{userCode}</div>}
      </div>

      <div className="card">
        <h2>导入凭证</h2>
        <p className="hint">粘贴或上传 Grok/xAI 的 xai 凭证 JSON（拒绝其它厂商文件名）。</p>
        <label>JSON 粘贴</label>
        <textarea
          value={jsonPaste}
          onChange={(e) => setJsonPaste(e.target.value)}
          placeholder='{"type":"xai", ...}'
        />
        <div className="row">
          <label className="check">
            <input
              type="checkbox"
              checked={importUsingApi}
              onChange={(e) => setImportUsingApi(e.target.checked)}
            />
            using_api
          </label>
          <button type="button" className="primary" onClick={() => void importPaste()}>
            粘贴导入
          </button>
        </div>
        <div className="row">
          <input ref={fileRef} type="file" accept=".json,application/json" />
          <button type="button" onClick={() => void importFile()}>
            文件导入
          </button>
        </div>
      </div>

      <div className="card">
        <h2>本地凭证文件</h2>
        {files.length === 0 ? (
          <p className="empty">暂无 xai-*.json</p>
        ) : (
          <div style={{ overflowX: "auto" }}>
            <table>
              <thead>
                <tr>
                  <th>厂商</th>
                  <th>文件</th>
                  <th>邮箱</th>
                  <th>过期</th>
                  <th>using_api</th>
                  <th />
                </tr>
              </thead>
              <tbody>
                {files.map((f) => (
                  <tr key={f.name || f.email}>
                    <td>{f.provider_label || f.provider || "Grok"}</td>
                    <td style={{ fontFamily: "var(--mono)", fontSize: "0.8rem" }}>{f.name}</td>
                    <td>{f.email || "—"}</td>
                    <td>{f.expired || "—"}</td>
                    <td>{f.using_api ? "true" : "false"}</td>
                    <td>
                      <button type="button" onClick={() => void selectFile(f.name || "")}>
                        设为当前
                      </button>
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
