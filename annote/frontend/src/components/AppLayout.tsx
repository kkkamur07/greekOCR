import { Button } from 'antd';
import { Link, useNavigate } from 'react-router-dom';
import { clearAccessToken } from '../auth/storage';

export function AppLayout({
  title,
  children,
  extra,
}: {
  title: string;
  children: React.ReactNode;
  extra?: React.ReactNode;
}) {
  const navigate = useNavigate();

  const handleLogout = () => {
    clearAccessToken();
    navigate('/login');
  };

  return (
    <div className="app-page">
      <header className="app-topnav">
        <Link to="/projects" className="app-topnav__logo" aria-label="nomicous home">
          <img src="/nomos.svg" alt="" />
        </Link>
        <div className="app-topnav__spacer" />
        {extra}
        <Button type="text" onClick={handleLogout} style={{ color: 'var(--text-3)' }}>
          Log out
        </Button>
      </header>
      <main className="app-content">
        <h1>{title}</h1>
        {children}
      </main>
    </div>
  );
}
