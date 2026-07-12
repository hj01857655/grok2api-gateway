import { NavLink, Outlet } from "react-router-dom";

const links = [
  { to: "/", end: true, label: "总览" },
  { to: "/accounts", label: "账号" },
  { to: "/models", label: "模型" },
  { to: "/logs", label: "请求日志" },
  { to: "/settings", label: "设置" },
];

export function Layout() {
  return (
    <div className="layout">
      <aside className="sidebar">
        <div className="brand">
          <h1>Grok2API</h1>
          <p>Admin · 官方 Grok</p>
        </div>
        <nav className="nav">
          {links.map((l) => (
            <NavLink
              key={l.to}
              to={l.to}
              end={l.end}
              className={({ isActive }) => (isActive ? "active" : undefined)}
            >
              {l.label}
            </NavLink>
          ))}
        </nav>
      </aside>
      <main className="main">
        <Outlet />
      </main>
    </div>
  );
}
