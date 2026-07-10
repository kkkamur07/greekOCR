"use client";

import { ConfigProvider } from "antd";
import type { ReactNode } from "react";
import { AuthProvider } from "../auth/AuthProvider";
import { BackgroundJobsPanel } from "../components/BackgroundJobsPanel";
import { ToastProvider } from "../components/ui/ToastProvider";
import { BackgroundJobsProvider } from "../context/BackgroundJobsContext";

export function Providers({ children }: { children: ReactNode }) {
  return (
    <ConfigProvider
      theme={{
        token: {
          colorPrimary: "#b40000",
          colorSuccess: "#059669",
          colorWarning: "#d97706",
          colorError: "#dc2626",
          colorLink: "#44403c",
          fontFamily: "'Inter', system-ui, -apple-system, sans-serif",
          borderRadius: 5,
        },
        components: {
          Layout: {
            headerBg: "#ffffff",
            bodyBg: "#faf9f7",
          },
        },
      }}
    >
      <AuthProvider>
        <ToastProvider>
          <BackgroundJobsProvider>
            {children}
            <BackgroundJobsPanel />
          </BackgroundJobsProvider>
        </ToastProvider>
      </AuthProvider>
    </ConfigProvider>
  );
}
