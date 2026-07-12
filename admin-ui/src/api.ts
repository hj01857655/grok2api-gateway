const LS_KEY = "g2a_admin_key";

export function getAdminKey(): string {
  return (localStorage.getItem(LS_KEY) || "").trim();
}

export function setAdminKey(key: string): void {
  localStorage.setItem(LS_KEY, key.trim());
}

export type ApiOpts = {
  method?: string;
  json?: unknown;
  headers?: Record<string, string>;
  body?: BodyInit;
};

export async function api<T = unknown>(path: string, opts: ApiOpts = {}): Promise<T> {
  const headers: Record<string, string> = { ...(opts.headers || {}) };
  const key = getAdminKey();
  if (key) {
    headers.Authorization = `Bearer ${key}`;
    headers["x-admin-key"] = key;
  }
  let body = opts.body;
  if (opts.json !== undefined) {
    headers["Content-Type"] = "application/json";
    body = JSON.stringify(opts.json);
  }
  const res = await fetch(path, { method: opts.method || "GET", headers, body });
  const ct = res.headers.get("content-type") || "";
  let data: unknown = null;
  if (ct.includes("application/json")) {
    data = await res.json();
  } else {
    data = { detail: await res.text() };
  }
  if (!res.ok) {
    const d = data as { detail?: unknown; message?: string };
    const detail = d?.detail;
    let msg: string;
    if (typeof detail === "string") msg = detail;
    else if (detail && typeof detail === "object" && "error" in (detail as object)) {
      const err = (detail as { error?: { message?: string } }).error;
      msg = err?.message || JSON.stringify(detail);
    } else if (d?.message) msg = d.message;
    else msg = res.statusText || `HTTP ${res.status}`;
    throw new Error(typeof msg === "string" ? msg : JSON.stringify(msg));
  }
  return data as T;
}

export type OAuthCurrent = {
  email?: string;
  sub?: string;
  expired?: string;
  using_api?: boolean;
  chat_base?: string;
  has_access?: boolean;
  has_refresh?: boolean;
  provider_label?: string;
};

export type OAuthFile = {
  name?: string;
  email?: string;
  expired?: string;
  using_api?: boolean;
  provider?: string;
  provider_label?: string;
};

export type Status = {
  ok?: boolean;
  version?: string;
  upstream_mode?: string;
  effective_upstream_mode?: string;
  upstream_key_configured?: boolean;
  oauth_auths_dir?: string;
  oauth_current?: OAuthCurrent | null;
  oauth_files?: OAuthFile[];
  admin_auth_required?: boolean;
  notes?: string[];
};

export type LogItem = {
  ts?: string;
  method?: string;
  path?: string;
  status?: number;
  duration_ms?: number;
  model?: string;
  stream?: boolean;
  error?: string;
  client?: string;
};

export type LogSummary = {
  ok?: boolean;
  last_1h?: { count: number; "4xx": number; "5xx": number };
  last_24h?: { count: number; "4xx": number; "5xx": number };
  logs_dir?: string;
};
