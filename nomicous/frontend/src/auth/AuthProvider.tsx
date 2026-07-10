import { createContext, useContext, useEffect, useMemo, useState, type ReactNode } from 'react';

import { api, refreshAccessToken } from '../api/client';
import { clearLoginRedirectGuard } from './session';
import { clearAccessToken, getAccessToken, setAccessToken } from './storage';

type AuthStatus = 'restoring' | 'authenticated' | 'anonymous';

type AuthContextValue = {
  status: AuthStatus;
  establish: (accessToken: string) => void;
  logout: () => Promise<void>;
};

const fallbackContext: AuthContextValue = {
  status: getAccessToken() ? 'authenticated' : 'anonymous',
  establish: (accessToken) => {
    setAccessToken(accessToken);
    clearLoginRedirectGuard();
  },
  logout: async () => {
    clearAccessToken();
  },
};

const AuthContext = createContext<AuthContextValue>(fallbackContext);

export function AuthProvider({ children }: { children: ReactNode }) {
  const [status, setStatus] = useState<AuthStatus>(
    getAccessToken() ? 'authenticated' : 'restoring',
  );

  useEffect(() => {
    if (getAccessToken()) return;
    let active = true;
    void refreshAccessToken()
      .then(() => {
        if (!active) return;
        clearLoginRedirectGuard();
        setStatus('authenticated');
      })
      .catch(() => {
        if (!active) return;
        clearAccessToken();
        setStatus('anonymous');
      });
    return () => {
      active = false;
    };
  }, []);

  const value = useMemo<AuthContextValue>(
    () => ({
      status,
      establish: (accessToken) => {
        setAccessToken(accessToken);
        clearLoginRedirectGuard();
        setStatus('authenticated');
      },
      logout: async () => {
        try {
          await api.logout();
        } finally {
          clearAccessToken();
          setStatus('anonymous');
        }
      },
    }),
    [status],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuthSession(): AuthContextValue {
  return useContext(AuthContext);
}
