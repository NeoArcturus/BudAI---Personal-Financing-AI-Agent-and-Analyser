"use client";

import React from "react";
import { Button, Skeleton } from "@heroui/react";
import {
  CheckCircle2,
  AlertTriangle,
  Info,
  Clock,
  ArrowRight,
} from "lucide-react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { toast } from "@heroui/react";

interface AlertItem {
  id: string;
  title?: string;
  message: string;
  event_type: string;
  urgency_level: number;
  status_code?: string;
  action_type?: string;
  action_payload?: any;
  is_read: boolean;
  is_dismissed: boolean;
  timestamp: string;
}

interface AutomatedActivityPanelProps {
  currency?: string;
  initialData?: AlertItem[];
}

export const AutomatedActivityPanel = ({
  currency = "GBP",
  initialData = [],
}: AutomatedActivityPanelProps) => {
  const queryClient = useQueryClient();

  const { data: alerts = initialData, isLoading } = useQuery({
    queryKey: ["notifications"],
    queryFn: async () => {
      const res = await apiFetch("/api/notifications?limit=10", {}, true);
      if (res.ok) {
        const data = (await res.json()) as AlertItem[];
        return data || [];
      }
      return initialData;
    },
    initialData: initialData.length > 0 ? initialData : undefined,
    staleTime: 1000 * 60 * 5,
    refetchInterval: false,
  });

  const handleDismiss = async (id: string) => {
    try {
      await apiFetch(`/api/notifications/${id}/dismiss`, { method: "POST" }, true);
      queryClient.invalidateQueries({ queryKey: ["notifications"] });
      toast("Notice dismissed.");
    } catch {
      toast("Unable to dismiss notice.", { variant: "danger" } as any);
    }
  };

  const handleApproveAction = async (alert: AlertItem) => {
    try {
      if (alert.action_type === "TRANSFER" && alert.action_payload) {
        const res = await apiFetch("/api/buckets/transfer", {
          method: "POST",
          body: JSON.stringify(alert.action_payload),
        }, true);
        if (res.ok) {
          toast("Transfer executed successfully.");
          await handleDismiss(alert.id);
          queryClient.invalidateQueries({ queryKey: ["buckets"] });
          return;
        }
      }
      toast("Action approved and recorded.");
      await handleDismiss(alert.id);
    } catch {
      toast("Failed to approve action.", { variant: "danger" } as any);
    }
  };

  return (
    <div className="flex flex-col gap-4 w-full">
      <div className="pb-1">
        <h2 className="text-base font-bold text-foreground">
          Activity & Alerts
        </h2>
        <p className="text-xs text-muted-foreground mt-0.5">
          Rule-based rebalancing and account notifications
        </p>
      </div>

      {/* Activity Feed */}
      <div className="flex flex-col gap-3">
        <span className="text-xs font-bold uppercase tracking-wider text-muted-foreground">
          Recent Activity
        </span>

        {isLoading ? (
          <div className="flex flex-col gap-2.5">
            {[1, 2].map((i) => (
              <div
                key={i}
                className="p-4 rounded-2xl border border-white/10 bg-white/30 dark:bg-neutral-900/30 backdrop-blur-xl flex flex-col gap-2"
              >
                <Skeleton className="h-4 w-3/4 rounded-md" />
                <Skeleton className="h-3 w-full rounded-md" />
              </div>
            ))}
          </div>
        ) : alerts.length === 0 ? (
          <div className="p-6 text-center rounded-2xl border border-dashed border-white/15 bg-white/40 dark:bg-neutral-900/30 backdrop-blur-xl flex flex-col items-center gap-2.5">
            <CheckCircle2 size={24} className="text-emerald-500/80" />
            <p className="text-sm font-semibold text-foreground">
              Everything in order
            </p>
            <p className="text-xs text-muted-foreground max-w-xs">
              Your financial rules and categories are monitored continuously.
            </p>
          </div>
        ) : (
          alerts.map((alert) => {
            const isHighUrgency = alert.urgency_level >= 3;
            const hasAction = !!alert.action_type;

            return (
              <div
                key={alert.id}
                className={`p-4 rounded-2xl border transition-colors text-xs flex flex-col gap-2.5 shadow-xs ${
                  isHighUrgency
                    ? "bg-amber-500/10 border-amber-500/30"
                    : "bg-[#0d0e14]/40 backdrop-blur-xl border-white/10 hover:border-white/20"
                }`}
              >
                <div className="flex items-start justify-between gap-2.5">
                  <div className="flex items-center gap-2 min-w-0">
                    {isHighUrgency ? (
                      <AlertTriangle
                        size={15}
                        className="text-amber-500 shrink-0 mt-0.5"
                      />
                    ) : (
                      <Info
                        size={15}
                        className="text-primary shrink-0 mt-0.5"
                      />
                    )}
                    <span className="font-bold text-foreground truncate text-xs md:text-sm">
                      {alert.title || "Account Notice"}
                    </span>
                  </div>

                  <span className="text-[10px] text-muted-foreground shrink-0 font-mono">
                    {new Date(alert.timestamp).toLocaleTimeString([], {
                      hour: "2-digit",
                      minute: "2-digit",
                    })}
                  </span>
                </div>

                <p className="text-xs text-muted-foreground leading-relaxed">
                  {alert.message}
                </p>

                <div className="flex items-center justify-between pt-2 border-t border-white/10 dark:border-white/5 text-[11px]">
                  <span className="text-muted-foreground capitalize font-medium">
                    {alert.event_type.replace(/_/g, " ").toLowerCase()}
                  </span>

                  <div className="flex items-center gap-1.5">
                    {hasAction && (
                      <Button
                        size="sm"
                        onPress={() => handleApproveAction(alert)}
                        className="h-7 px-3 text-xs font-bold rounded-lg bg-primary text-primary-foreground hover:opacity-90 shadow-xs"
                      >
                        Approve
                      </Button>
                    )}
                    <button
                      onClick={() => handleDismiss(alert.id)}
                      className="text-muted-foreground hover:text-foreground px-2 py-1 rounded-md hover:bg-foreground/5 cursor-pointer transition-colors"
                    >
                      Dismiss
                    </button>
                  </div>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
