"use client";
import { useEffect } from "react";

export function GlobalErrorReporter() {
  useEffect(() => {
    const post = (payload: Record<string, unknown>) =>
      fetch("/api/log", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      }).catch(() => {});

    const onError = (event: ErrorEvent) => {
      post({
        type: "onerror",
        message: event.message,
        source: event.filename,
        lineno: event.lineno,
        colno: event.colno,
        stack: event.error?.stack ?? null,
      });
    };

    const onUnhandled = (event: PromiseRejectionEvent) => {
      post({
        type: "unhandledrejection",
        message: String(event.reason),
        stack: (event.reason as { stack?: string } | null)?.stack ?? null,
      });
    };

    window.addEventListener("error", onError);
    window.addEventListener("unhandledrejection", onUnhandled);
    return () => {
      window.removeEventListener("error", onError);
      window.removeEventListener("unhandledrejection", onUnhandled);
    };
  }, []);
  return null;
}
