import { useEffect, useState } from "react";
import { reportClientFailure } from "../../api/failureBeacon";
import { registerToastHandler, type ToastItem } from "./toast";

const DISMISS_MS = 2800;

export function ToastProvider({ children }: { children: React.ReactNode }) {
  const [toasts, setToasts] = useState<ToastItem[]>([]);

  useEffect(() => {
    registerToastHandler(({ message, variant }) => {
      const id = crypto.randomUUID();
      setToasts((prev) => [...prev, { id, message, variant }]);
      if (variant === "error") {
        reportClientFailure(new Error(message), "toast");
      }
      window.setTimeout(() => {
        setToasts((prev) => prev.filter((t) => t.id !== id));
      }, DISMISS_MS);
    });
    return () => registerToastHandler(null);
  }, []);

  return (
    <>
      {children}
      <div className="toast-stack" aria-live="polite" aria-relevant="additions">
        {toasts.map((item) => (
          <div
            key={item.id}
            className={`toast toast--${item.variant}`}
            role={item.variant === "error" ? "alert" : "status"}
          >
            {item.message}
          </div>
        ))}
      </div>
    </>
  );
}
