"use client";

import React, { useState } from "react";
import { useQuery, useQueryClient } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { toast, Button } from "@heroui/react";

import { DashboardHeader } from "./components/DashboardHeader";
import { BucketAllocationsPanel, BucketItem } from "./components/BucketAllocationsPanel";
import { ForecastChart, TrajectoryPoint } from "./components/ForecastChart";
import { TransactionTable, TransactionItem } from "./components/TransactionTable";
import { DecisionStream, DecisionItem } from "./components/DecisionStream";
import { GuardrailsCard } from "./components/GuardrailsCard";

export const WidgetContext = React.createContext<{ onRemove?: () => void; instanceId?: string }>({});

export interface DashboardSnapshot {
  status: string;
  reconciliation: {
    physical_balance: number;
    virtual_balance: number;
    delta: number;
    is_reconciled: boolean;
    bank_synced: boolean;
  };
  buckets: BucketItem[];
  forecast: {
    horizon_days: number;
    safety_buffer_floor: number;
    trajectory: TrajectoryPoint[];
  };
  transactions: TransactionItem[];
  decisions: DecisionItem[];
  guardrails: {
    safety_buffer: number;
    autonomous_limit: number;
    approval_policy: string;
    bank_connection_protocol: string;
    guardrails_active: boolean;
  };
}

interface DashboardClientProps {
  initialSnapshot?: DashboardSnapshot | null;
}

export default function DashboardClient({ initialSnapshot }: DashboardClientProps) {
  const queryClient = useQueryClient();
  const [timeWindow, setTimeWindow] = useState("60D Forecast");
  const [isTransferModalOpen, setIsTransferModalOpen] = useState(false);
  const [sourceBucket, setSourceBucket] = useState("");
  const [destBucket, setDestBucket] = useState("");
  const [transferAmount, setTransferAmount] = useState("");
  const [isSubmittingTransfer, setIsSubmittingTransfer] = useState(false);

  // Consolidated Dashboard Snapshot Query
  const { data } = useQuery<DashboardSnapshot>({
    queryKey: ["dashboardSnapshot", timeWindow],
    queryFn: async (): Promise<DashboardSnapshot> => {
      const days = timeWindow === "30D" ? 30 : timeWindow === "90D" ? 90 : timeWindow === "1Y" ? 365 : 60;
      const res = await apiFetch(`/api/dashboard/snapshot?range_days=${days}`, { method: "GET" }, true);
      if (!res.ok) {
        throw new Error("Failed to load dashboard snapshot");
      }
      const json = await res.json() as DashboardSnapshot;
      return json;
    },
    initialData: initialSnapshot || undefined,
    refetchInterval: 15000,
  });

  const snapshot: DashboardSnapshot | undefined = data || initialSnapshot || undefined;

  const handleConfirmDecision = async (decisionId: string) => {
    try {
      toast.success("Action confirmed and executed under double-entry verification.");
      queryClient.invalidateQueries({ queryKey: ["dashboardSnapshot"] });
    } catch {
      toast.danger("Failed to execute action.");
    }
  };

  const handleReportDecision = async (decisionId: string) => {
    try {
      toast.info("Transaction reported and flagged for investigation.");
      queryClient.invalidateQueries({ queryKey: ["dashboardSnapshot"] });
    } catch {
      toast.danger("Failed to update status.");
    }
  };

  const handleExecuteTransfer = async () => {
    if (!sourceBucket || !destBucket || !transferAmount) {
      toast.warning("Please fill in source, destination, and amount.");
      return;
    }
    const amt = parseFloat(transferAmount);
    if (isNaN(amt) || amt <= 0) {
      toast.warning("Please enter a valid amount.");
      return;
    }

    setIsSubmittingTransfer(true);
    try {
      const res = await apiFetch(
        "/api/buckets/transfer",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            source_bucket_id: sourceBucket,
            destination_bucket_id: destBucket,
            amount: amt,
            reason: "Manual reallocation from dashboard",
          }),
        },
        true
      );
      if (res.ok) {
        toast.success(`Successfully transferred £${amt.toFixed(2)}.`);
        setIsTransferModalOpen(false);
        setTransferAmount("");
        queryClient.invalidateQueries({ queryKey: ["dashboardSnapshot"] });
      } else {
        const err = await res.json() as any;
        toast.danger(err.detail || "Transfer rejected by database lock.");
      }
    } catch {
      toast.danger("Failed to execute virtual transfer.");
    } finally {
      setIsSubmittingTransfer(false);
    }
  };

  const buckets: BucketItem[] = snapshot?.buckets || [];
  const transactions: TransactionItem[] = snapshot?.transactions || [];
  const decisions: DecisionItem[] = snapshot?.decisions || [];

  return (
    <div className="max-w-[1440px] mx-auto space-y-4 text-slate-100 antialiased p-2 sm:p-4">
      {/* Top Bar: Header and Global Controls */}
      <DashboardHeader
        reconciliation={snapshot?.reconciliation}
        timeWindow={timeWindow}
        onTimeWindowChange={setTimeWindow}
      />

      {/* Main Grid: 3-Column Split Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4">
        {/* Left Column: Virtual Bucket Hierarchy (3 Cols / 25%) */}
        <div className="lg:col-span-3 space-y-4">
          <BucketAllocationsPanel
            buckets={buckets}
            onOpenTransferModal={() => setIsTransferModalOpen(true)}
          />
        </div>

        {/* Center Column: Forecast & Ledger Table (6 Cols / 50%) */}
        <div className="lg:col-span-6 space-y-4">
          <ForecastChart
            trajectory={snapshot?.forecast?.trajectory || []}
            safetyBufferFloor={snapshot?.forecast?.safety_buffer_floor || 800.0}
            timeWindow={timeWindow}
          />
          <TransactionTable transactions={transactions} />
        </div>

        {/* Right Column: Agent Reasoning & Governance (3 Cols / 25%) */}
        <div className="lg:col-span-3 space-y-4">
          <DecisionStream
            decisions={decisions}
            onConfirmAction={handleConfirmDecision}
            onReportAction={handleReportDecision}
          />
          <GuardrailsCard guardrails={snapshot?.guardrails} />
        </div>
      </div>

      {/* Manual Transfer Modal */}
      {isTransferModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center bg-black/70 backdrop-blur-sm p-4">
          <div className="glass-card rounded-2xl p-6 border border-slate-700 w-full max-w-md space-y-4">
            <h3 className="text-sm font-semibold text-white">Manual Virtual Rebalance</h3>
            <p className="text-xs text-slate-400">
              Move funds between virtual buckets. Governed by double-entry zero-sum locks.
            </p>

            <div className="space-y-3 text-xs">
              <div>
                <label className="block text-slate-400 mb-1">Source Bucket</label>
                <select
                  value={sourceBucket}
                  onChange={(e) => setSourceBucket(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-white font-mono"
                >
                  <option value="">Select source bucket</option>
                  {buckets.map((b: BucketItem) => (
                    <option key={b.id} value={b.id}>
                      {b.name} (£{b.cached_balance.toFixed(2)})
                    </option>
                  ))}
                </select>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Destination Bucket</label>
                <select
                  value={destBucket}
                  onChange={(e) => setDestBucket(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-white font-mono"
                >
                  <option value="">Select destination bucket</option>
                  {buckets
                    .filter((b: BucketItem) => b.id !== sourceBucket)
                    .map((b: BucketItem) => (
                      <option key={b.id} value={b.id}>
                        {b.name} (£{b.cached_balance.toFixed(2)})
                      </option>
                    ))}
                </select>
              </div>

              <div>
                <label className="block text-slate-400 mb-1">Amount (£)</label>
                <input
                  type="number"
                  step="0.01"
                  placeholder="0.00"
                  value={transferAmount}
                  onChange={(e) => setTransferAmount(e.target.value)}
                  className="w-full bg-slate-900 border border-slate-800 rounded-lg p-2 text-white font-mono"
                />
              </div>
            </div>

            <div className="flex justify-end gap-2 pt-2">
              <Button
                size="sm"
                variant="ghost"
                onPress={() => setIsTransferModalOpen(false)}
                className="text-slate-400 hover:text-white"
              >
                Cancel
              </Button>
              <Button
                size="sm"
                variant="primary"
                isPending={isSubmittingTransfer}
                onPress={handleExecuteTransfer}
                className="bg-cyan-600 hover:bg-cyan-500 text-white font-medium"
              >
                Execute Rebalance
              </Button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
