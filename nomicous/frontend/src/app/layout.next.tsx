import type { Metadata } from 'next';
import type { ReactNode } from 'react';
import '../index.css';
import '../styles/nomicous.css';
import { Providers } from './providers';

export const metadata: Metadata = {
  title: 'Nomicous',
  description: 'Nomicous annotation platform',
};

export default function RootLayout({ children }: Readonly<{ children: ReactNode }>) {
  return (
    <html lang="en">
      <body>
        <Providers>{children}</Providers>
      </body>
    </html>
  );
}
