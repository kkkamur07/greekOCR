"use client";

import { QueryClientProvider } from "@tanstack/react-query";
import { App, ConfigProvider } from "antd";
import type { ReactNode } from "react";
import { queryClient } from "../api/queryClient";
import { AuthProvider } from "../auth/AuthProvider";
import { BackgroundJobsPanel } from "../components/BackgroundJobsPanel";
import { ToastBridge } from "../components/ui/ToastBridge";
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
      {/* `App` supplies the themed `message` instance the `toast` helper uses. */}
      <App>
        <ToastBridge />
        <QueryClientProvider client={queryClient}>
          <AuthProvider>
            <BackgroundJobsProvider>
              {children}
              <BackgroundJobsPanel />
            </BackgroundJobsProvider>
          </AuthProvider>
        </QueryClientProvider>
      </App>
    </ConfigProvider>
  );
}
