"use client";

import { App } from "antd";
import { useEffect } from "react";
import { registerToastApi } from "./toast";

/**
 * Hands antd's theme-bound `message` instance to the module-level `toast`
 * helper. Renders nothing; `App` itself supplies the container, the stacking,
 * the dismiss timer, and the `aria-live` region.
 */
export function ToastBridge() {
  const { message } = App.useApp();

  useEffect(() => {
    registerToastApi(message);
    return () => registerToastApi(null);
  }, [message]);

  return null;
}
