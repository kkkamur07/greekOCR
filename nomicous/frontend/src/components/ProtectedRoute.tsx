import { useEffect } from 'react';
import { useLocation, useNavigate } from 'react-router-dom';
import { hasAccessToken, navigateToLogin } from '../auth/session';

export function ProtectedRoute({ children }: { children: React.ReactNode }) {
  const location = useLocation();
  const navigate = useNavigate();
  const authed = hasAccessToken();

  useEffect(() => {
    if (!authed) {
      navigateToLogin(navigate, location);
    }
  }, [authed, location, navigate]);

  if (!authed) {
    return null;
  }

  return <>{children}</>;
}
