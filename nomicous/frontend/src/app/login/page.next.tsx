'use client';

import { Suspense } from 'react';
import { LoginPage } from '../../pages/LoginPage';

export default function LoginRoute() {
  return (
    <Suspense fallback={null}>
      <LoginPage />
    </Suspense>
  );
}
