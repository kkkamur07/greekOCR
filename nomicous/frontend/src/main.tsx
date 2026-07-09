import { StrictMode } from 'react';
import { createRoot } from 'react-dom/client';
import { BrowserRouter } from 'react-router-dom';
import { ConfigProvider } from 'antd';
import App from './App.tsx';
import { BackgroundJobsPanel } from './components/BackgroundJobsPanel.tsx';
import { ToastProvider } from './components/ui/ToastProvider.tsx';
import { BackgroundJobsProvider } from './context/BackgroundJobsContext.tsx';
import './index.css';
import './styles/nomicous.css';

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: '#b40000',
          colorSuccess: '#059669',
          colorWarning: '#d97706',
          colorError: '#dc2626',
          colorLink: '#44403c',
          fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
          borderRadius: 5,
        },
        components: {
          Layout: {
            headerBg: '#ffffff',
            bodyBg: '#faf9f7',
          },
        },
      }}
    >
      <BrowserRouter>
        <ToastProvider>
          <BackgroundJobsProvider>
            <App />
            <BackgroundJobsPanel />
          </BackgroundJobsProvider>
        </ToastProvider>
      </BrowserRouter>
    </ConfigProvider>
  </StrictMode>,
);
