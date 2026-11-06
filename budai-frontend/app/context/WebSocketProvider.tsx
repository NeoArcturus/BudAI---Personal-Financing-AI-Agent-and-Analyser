"use client";

import React, { createContext, useContext, useEffect, useRef, useState, useCallback } from "react";
import { toast } from "@heroui/react";
import { useQueryClient } from "@tanstack/react-query";
import { getAuthToken, getApiBaseUrl } from "@/lib/api";

interface WebSocketContextType {
  isConnected: boolean;
  sendMessage: (msg: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType>({
  isConnected: false,
  sendMessage: () => {},
});

export const useBudAIWebSocket = () => useContext(WebSocketContext);

const BUCKET_EVENTS = new Set([
  "BUCKET_BALANCES_UPDATED",
  "VIRTUAL_TRANSFER",
  "BUCKET_UPDATED",
  "BUCKET_DELETED",
  "BUCKET_CREATED",
]);

export const WebSocketProvider = ({ children }: { children: React.ReactNode }) => {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const queryClient = useQueryClient();

  // Debounced batch query invalidation queue (200ms settlement window)
  const pendingKeysRef = useRef<Set<string>>(new Set());
  const debounceTimerRef = useRef<NodeJS.Timeout | null>(null);

  const queueInvalidation = useCallback((keys: string[]) => {
    keys.forEach((k) => pendingKeysRef.current.add(k));

    if (debounceTimerRef.current) {
      clearTimeout(debounceTimerRef.current);
    }

    debounceTimerRef.current = setTimeout(() => {
      const keysToInvalidate = Array.from(pendingKeysRef.current);
      pendingKeysRef.current.clear();
      debounceTimerRef.current = null;

      keysToInvalidate.forEach((key) => {
        queryClient.invalidateQueries({ queryKey: [key] });
      });
    }, 200);
  }, [queryClient]);

  useEffect(() => {
    let reconnectTimeout: NodeJS.Timeout | null = null;
    let retryAttempt = 0;
    let isDisposed = false;

    const connect = () => {
      if (isDisposed) return;

      const currentToken = getAuthToken();
      if (!currentToken) {
        setIsConnected(false);
        // Incremental retry loop (1.5s, 3s, 5s cap) waiting for auth hydration
        const delay = Math.min(1500 * Math.pow(1.5, retryAttempt), 5000);
        retryAttempt += 1;
        reconnectTimeout = setTimeout(connect, delay);
        return;
      }

      // Reset retry counter once token exists
      retryAttempt = 0;

      // Close previous instance if still hanging
      if (wsRef.current && (wsRef.current.readyState === WebSocket.OPEN || wsRef.current.readyState === WebSocket.CONNECTING)) {
        return;
      }

      const API_BASE_URL = getApiBaseUrl();
      const freshWsUrl = `${API_BASE_URL.replace(/^http/, "ws")}/ws?token=${encodeURIComponent(currentToken)}`;

      try {
        const ws = new WebSocket(freshWsUrl);

        ws.onopen = () => {
          if (!isDisposed) {
            setIsConnected(true);
          }
        };

        ws.onmessage = (event) => {
          try {
            const payload = JSON.parse(event.data);
            const { event_type } = payload;

            if (event_type === "NEW_ALERT") {
              const alertData = payload.alert || payload.data || payload;
              const message = alertData.message || payload.message || "New system alert";
              const action = alertData.action || alertData.action_type || payload.action;
              const isUrgent =
                action === "REAUTH_TRUELAYER" ||
                alertData.urgency === "HIGH" ||
                alertData.priority === "HIGH";

              toast(message, {
                variant: isUrgent ? "danger" : "default",
                actionProps: action
                  ? {
                      children: "Resolve",
                      onPress: () => {
                        if (action === "REAUTH_TRUELAYER") {
                          window.location.href = "/connections";
                        }
                      },
                    }
                  : undefined,
              });

              // Instantly prepend alert into the TanStack Query cache (Zero HTTP roundtrips)
              queryClient.setQueryData<any[]>(["notifications"], (old = []) => {
                if (!Array.isArray(old)) return [alertData];
                if (
                  old.some(
                    (item) =>
                      (item.id && item.id === alertData.id) ||
                      (item.alert_uuid && item.alert_uuid === alertData.alert_uuid),
                  )
                ) {
                  return old;
                }
                return [alertData, ...old];
              });

              // Invalidate auxiliary alert keys without forcing redundant GET /api/notifications
              queueInvalidation(["systemAlerts"]);
            } else if (BUCKET_EVENTS.has(event_type)) {
              queueInvalidation(["buckets", "accounts", "notifications"]);
            } else if (event_type === "TRANSACTIONS_UPDATED") {
              queueInvalidation(["transactions", "accounts", "spending-trends", "expense-categories"]);
            }
          } catch (err) {
            console.error("Failed to parse WS message", err);
          }
        };

        ws.onclose = () => {
          if (!isDisposed) {
            setIsConnected(false);
            reconnectTimeout = setTimeout(connect, 3000);
          }
        };

        ws.onerror = () => {
          console.warn("WebSocket connection unavailable, retrying in background...");
          ws.close();
        };

        wsRef.current = ws;
      } catch (e) {
        reconnectTimeout = setTimeout(connect, 3000);
      }
    };

    // Reactive handler for authentication changes (same-tab and cross-tab)
    const handleAuthChange = () => {
      const token = getAuthToken();
      if (token) {
        if (!wsRef.current || wsRef.current.readyState === WebSocket.CLOSED) {
          if (reconnectTimeout) clearTimeout(reconnectTimeout);
          connect();
        }
      } else {
        if (wsRef.current) {
          wsRef.current.close();
        }
        setIsConnected(false);
      }
    };

    window.addEventListener("storage", handleAuthChange);
    window.addEventListener("budai-auth-change", handleAuthChange);

    connect();

    return () => {
      isDisposed = true;
      if (reconnectTimeout) clearTimeout(reconnectTimeout);
      if (debounceTimerRef.current) clearTimeout(debounceTimerRef.current);
      window.removeEventListener("storage", handleAuthChange);
      window.removeEventListener("budai-auth-change", handleAuthChange);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [queueInvalidation]);

  const sendMessage = (msg: any) => {
    if (wsRef.current && wsRef.current.readyState === WebSocket.OPEN) {
      wsRef.current.send(JSON.stringify(msg));
    }
  };

  return (
    <WebSocketContext.Provider value={{ isConnected, sendMessage }}>
      {children}
    </WebSocketContext.Provider>
  );
};
