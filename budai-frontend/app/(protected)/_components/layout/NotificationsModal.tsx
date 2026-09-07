"use client";

import React, { useState } from "react";
import { Modal, Button, ScrollShadow } from "@heroui/react";
import { useQuery } from "@tanstack/react-query";
import { Bell, AlertTriangle, ShieldCheck, ArrowRightLeft, Info, X } from "lucide-react";
import { getAuthToken } from "@/lib/api";

interface NotificationsModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  initialAlerts?: any[];
}

export function NotificationsModal({ isOpen, onOpenChange, initialAlerts = [] }: NotificationsModalProps) {
  const { data: alerts } = useQuery({
    queryKey: ["systemAlerts"],
    queryFn: async () => {
      const token = getAuthToken();
      const API_BASE_URL = process.env.NEXT_PUBLIC_API_BASE_URL?.replace(/\/$/, "") || "http://localhost:8080";
      const res = await fetch(`${API_BASE_URL}/api/alerts/history`, {
        headers: { Authorization: `Bearer ${token}` }
      });
      if (!res.ok) return initialAlerts;
      return await res.json();
    },
    initialData: initialAlerts,
  });

  const getAlertIcon = (actionType?: string) => {
    switch (actionType) {
      case "REAUTH_TRUELAYER": return <AlertTriangle size={16} className="text-red-500" />;
      case "SWEEP": return <ArrowRightLeft size={16} className="text-green-500" />;
      case "SECURITY": return <ShieldCheck size={16} className="text-blue-500" />;
      default: return <Info size={16} className="text-foreground" />;
    }
  };

  const getAlertStyle = (actionType?: string) => {
    switch (actionType) {
      case "REAUTH_TRUELAYER": return "bg-red-500/5 border-red-500/20";
      case "SWEEP": return "bg-green-500/5 border-green-500/20";
      case "SECURITY": return "bg-blue-500/5 border-blue-500/20";
      default: return "bg-white/5 border-white/10";
    }
  };

  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={onOpenChange} variant="blur">
      <Modal.Container placement="center">
        <Modal.Dialog className="modal relative w-full max-w-2xl bg-black/90 border border-white/5 shadow-2xl rounded-2xl flex flex-col overflow-hidden pointer-events-auto">
          <button
            onClick={() => onOpenChange(false)}
            className="absolute top-6 right-6 z-10 w-8 h-8 rounded-full bg-white/5 hover:bg-white/10 text-white/50 hover:text-white flex items-center justify-center transition-colors"
          >
            <X size={16} />
          </button>

          <div className="flex flex-col gap-1 border-b border-white/5 p-8 bg-content1/50">
            <div className="flex items-center gap-4">
              <div className="w-10 h-10 rounded-full bg-primary/10 border border-primary/20 flex items-center justify-center">
                <Bell size={20} className="text-primary" />
              </div>
              <div>
                <h2 className="text-[10px] font-black uppercase tracking-[0.4em] italic text-primary">Notifications</h2>
                <p className="text-[11px] text-muted-foreground font-mono">System activity and alerts</p>
              </div>
            </div>
          </div>

          <div className="p-0">
            <ScrollShadow hideScrollBar className="w-full h-[60vh] p-8">
              <div className="flex flex-col gap-6 relative before:absolute before:inset-y-0 before:left-6 before:w-[1px] before:bg-white/5 before:z-0">
                {alerts?.length > 0 ? alerts.map((alert: any, idx: number) => (
                  <div key={alert.id || idx} className="relative z-10 flex gap-6">
                    <div className={`w-10 h-10 rounded-full border shrink-0 flex items-center justify-center bg-background ${getAlertStyle(alert.action)}`}>
                      {getAlertIcon(alert.action)}
                    </div>
                    <div className="flex flex-col pt-1 w-full">
                      <span className="text-[11px] font-mono text-muted-foreground mb-2">
                        {new Date(alert.created_at || Date.now()).toLocaleString("en-GB", {
                          hour: "2-digit", minute: "2-digit", day: "2-digit", month: "short"
                        })}
                      </span>
                      <div className={`p-5 rounded-2xl border ${getAlertStyle(alert.action)}`}>
                        <p className="text-sm text-foreground/90 leading-relaxed mb-3">{alert.message}</p>

                        {/* Actionable Smart Card Integration */}
                        {alert.action === "REAUTH_TRUELAYER" && (
                          <Button
                            size="sm"
                            variant="danger"
                            className="w-full font-bold tracking-widest uppercase text-[10px] mt-2"
                            onPress={() => window.location.href = "/connections"}
                          >
                            Re-Authenticate Now
                          </Button>
                        )}
                        {alert.action === "SWEEP" && (
                          <div className="flex items-center gap-3 mt-3 pt-3 border-t border-green-500/10">
                            <span className="text-[11px] font-mono text-green-500/70 uppercase tracking-widest">{alert.metadata?.source || "DEFAULT"}</span>
                            <ArrowRightLeft size={12} className="text-green-500/50" />
                            <span className="text-[11px] font-mono text-green-500/70 uppercase tracking-widest">{alert.metadata?.target || "ACCUMULATING"}</span>
                          </div>
                        )}
                      </div>
                    </div>
                  </div>
                )) : (
                  <div className="flex flex-col items-center justify-center h-48 text-muted-foreground z-10 bg-background/50 rounded-2xl border border-white/5">
                    <Bell size={24} className="opacity-20 mb-3" />
                    <p className="text-xs font-mono uppercase tracking-widest text-center px-8">No notifications available.</p>
                  </div>
                )}
              </div>
            </ScrollShadow>
          </div>
        </Modal.Dialog>
      </Modal.Container>
    </Modal.Backdrop>
  );
}
