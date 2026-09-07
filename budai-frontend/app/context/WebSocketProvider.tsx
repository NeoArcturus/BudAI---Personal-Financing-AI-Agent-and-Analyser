"use client";

import React, { createContext, useContext, useEffect, useRef, useState } from "react";
import { toast } from "@heroui/react";
import { useQueryClient } from "@tanstack/react-query";
import { getAuthToken } from "@/lib/api";

interface WebSocketContextType {
  isConnected: boolean;
  sendMessage: (msg: any) => void;
}

const WebSocketContext = createContext<WebSocketContextType>({
  isConnected: false,
  sendMessage: () => {},
});

export const useBudAIWebSocket = () => useContext(WebSocketContext);

export const WebSocketProvider = ({ children }: { children: React.ReactNode }) => {
  const [isConnected, setIsConnected] = useState(false);
  const wsRef = useRef<WebSocket | null>(null);
  const queryClient = useQueryClient();

  useEffect(() => {
    let reconnectTimeout: NodeJS.Timeout;

    const connect = () => {
      // Dynamically fetch the freshest token right before connecting
      // to ensure we don't close over an expired token during auto-reconnects
      const currentToken = getAuthToken();
      if (!currentToken) return;

      const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8080";
      const freshWsUrl = `${API_BASE_URL.replace(/^http/, "ws")}/ws?token=${currentToken}`;
      
      const ws = new WebSocket(freshWsUrl);
      
      ws.onopen = () => {
        setIsConnected(true);
      };

      ws.onmessage = (event) => {
        try {
          const payload = JSON.parse(event.data);
          const { event_type, data } = payload;

          if (event_type === "NEW_ALERT") {
            const isAlertUrgent = data?.action === "REAUTH_TRUELAYER";
            
            toast(data?.message || "New system alert", {
              variant: isAlertUrgent ? "danger" : "default",
              actionProps: data?.action ? {
                children: "Resolve",
                onPress: () => {
                  if (data.action === "REAUTH_TRUELAYER") {
                    window.location.href = "/connections";
                  }
                }
              } : undefined
            });
            queryClient.invalidateQueries({ queryKey: ["systemAlerts"] });
          } else if (event_type === "BUCKET_BALANCES_UPDATED") {
            queryClient.invalidateQueries({ queryKey: ["buckets"] });
            queryClient.invalidateQueries({ queryKey: ["accounts"] });
          } else if (event_type === "TRANSACTIONS_UPDATED") {
             queryClient.invalidateQueries({ queryKey: ["transactions"] });
          }
        } catch (err) {
          console.error("Failed to parse WS message", err);
        }
      };

      ws.onclose = () => {
        setIsConnected(false);
        reconnectTimeout = setTimeout(connect, 3000);
      };

      ws.onerror = (error) => {
        console.warn("WebSocket connection unavailable, retrying in background...");
        ws.close();
      };

      wsRef.current = ws;
    };

    connect();

    return () => {
      clearTimeout(reconnectTimeout);
      if (wsRef.current) {
        wsRef.current.close();
      }
    };
  }, [queryClient]);

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
