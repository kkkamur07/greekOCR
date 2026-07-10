import { useState, type FormEvent } from 'react';
import Link from 'next/link';
import { useRouter, useSearchParams } from 'next/navigation';
import { toast } from '../components/ui/toast';
import { api, type RegisterRequest } from '../api/client';
import { ApiError } from '../api/errors';
import { authRedirectTarget } from '../auth/redirect';
import { useAuthSession } from '../auth/AuthProvider';
import { AuthFormWrap, AuthLayout } from '../components/layout/AuthLayout';
import { PasswordInput } from '../components/ui/PasswordInput';

export function RegisterPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const [loading, setLoading] = useState(false);
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const from = authRedirectTarget(searchParams?.get('callbackUrl'));
  const { establish } = useAuthSession();

  const onSubmit = async (event: FormEvent<HTMLFormElement>) => {
    event.preventDefault();
    setLoading(true);
    const values: RegisterRequest = { email, username, password };
    try {
      const token = await api.register(values);
      establish(token.access_token);
      toast.success('Account created');
      router.replace(from);
    } catch (err) {
      const msg = err instanceof ApiError ? err.message : 'Registration failed';
      toast.error(msg);
    } finally {
      setLoading(false);
    }
  };

  return (
    <AuthLayout
      headline="Nomicous platform for HTR"
      tagline="Built for the NOMOS project. Upload pages, pair transcriptions to segments, pick the HTR model for each script, and share finished work as a live public page."
    >
      <AuthFormWrap>
        <h1>Register</h1>
        <p className="auth-sub">Create your account</p>
        <form onSubmit={onSubmit}>
          <div className="field">
            <label htmlFor="email">Email</label>
            <input
              id="email"
              type="email"
              placeholder="you@institution.edu"
              autoComplete="email"
              required
              value={email}
              onChange={(e) => setEmail(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="username">Username</label>
            <input
              id="username"
              type="text"
              placeholder="dr-smith"
              autoComplete="username"
              required
              value={username}
              onChange={(e) => setUsername(e.target.value)}
            />
          </div>
          <div className="field">
            <label htmlFor="password">
              Password <span className="text-muted" id="password-hint">(8+ chars)</span>
            </label>
            <PasswordInput
              id="password"
              autoComplete="new-password"
              aria-describedby="password-hint"
              minLength={8}
              value={password}
              onChange={(e) => setPassword(e.target.value)}
            />
          </div>
          <button type="submit" className="btn btn-primary btn-block mt-4" disabled={loading}>
            {loading ? 'Creating…' : 'Create account'}
          </button>
        </form>
        <p className="auth-footer-link">
          Have an account? <Link href={`/login?callbackUrl=${encodeURIComponent(from)}`}>Sign in</Link>
        </p>
      </AuthFormWrap>
    </AuthLayout>
  );
}
