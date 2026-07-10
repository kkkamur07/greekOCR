export type ToastVariant = "success" | "error";

export type ToastItem = {
  id: string;
  message: string;
  variant: ToastVariant;
};

type ToastListener = (item: Omit<ToastItem, "id">) => void;

let listener: ToastListener | null = null;

export function registerToastHandler(handler: ToastListener | null) {
  listener = handler;
}

function push(variant: ToastVariant, message: string) {
  listener?.({ message, variant });
}

export const toast = {
  success: (message: string) => push("success", message),
  error: (message: string) => push("error", message),
};
