"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  RefreshCcw,
  Plus,
  Database,
  ShieldCheck,
  Building2,
  Calendar,
  Clock,
  ArrowUpRight,
} from "lucide-react";
import { Button, Card, Chip, ProgressBar, toast } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";

export default function ConnectionsPage() {
  const queryClient = useQueryClient();
  const { accounts } = useBudAI();
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [isExtending, setIsExtending] = useState(false);

  useEffect(() => {
    if (accounts.length > 0 && !selectedAccountId) {
      setSelectedAccountId(accounts[0].account_id);
    }
  }, [accounts, selectedAccountId]);

  const selectedAccount = useMemo(() => {
    return accounts.find((a) => a.account_id === selectedAccountId) || accounts[0] || null;
  }, [accounts, selectedAccountId]);

  const handleConnectBank = async () => {
    setIsConnecting(true);
    try {
      const res = await apiFetch("/api/auth/banks/connect", { method: "POST" }, true);
      const data = await res.json() as any;
      if (data.auth_url || data.auth_uri) {
        window.location.href = data.auth_url || data.auth_uri;
      } else {
        toast.danger("Failed to initiate bank authorization link.");
      }
    } catch {
      toast.danger("Network error connecting to TrueLayer.");
    } finally {
      setIsConnecting(false);
    }
  };

  const handleSyncAccount = async () => {
    if (!selectedAccount?.bank_uuid) return;
    setIsSyncing(true);
    try {
      const res = await apiFetch(`/api/accounts/banks/${selectedAccount.bank_uuid}/sync`, { method: "POST" }, true);
      if (res.ok) {
        toast.success("Account statement synchronized successfully.");
        queryClient.invalidateQueries({ queryKey: ["accounts"] });
        queryClient.invalidateQueries({ queryKey: ["dashboardSnapshot"] });
      } else {
        toast.danger("Sync request failed.");
      }
    } catch {
      toast.danger("Network error syncing account.");
    } finally {
      setIsSyncing(false);
    }
  };

  const handleExtendConsent = async () => {
    if (!selectedAccount?.bank_uuid) return;
    setIsExtending(true);
    try {
      const res = await apiFetch(`/api/accounts/banks/${selectedAccount.bank_uuid}/extend`, { method: "POST" }, true);
      const data = await res.json() as any;
      if (data.status === "extended") {
        toast.success("Open Banking consent extended for 90 days.");
        queryClient.invalidateQueries({ queryKey: ["accounts"] });
      } else if (data.status === "reauth_required") {
        toast.warning("Re-authorization required. Redirecting...");
        const linkRes = await apiFetch(`/api/auth/banks/${selectedAccount.bank_uuid}/reauth`, { method: "POST" }, true);
        const linkData = await linkRes.json() as any;
        if (linkData.auth_url || linkData.auth_uri) {
          window.location.href = linkData.auth_url || linkData.auth_uri;
        }
      }
    } catch {
      toast.danger("Failed to extend consent.");
    } finally {
      setIsExtending(false);
    }
  };

  return (
    <div className="w-full flex-1 flex flex-col h-full bg-[#0b0f17] text-slate-100 font-sans p-4 sm:p-6 space-y-4">
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-800">
        <div>
          <h1 className="text-base font-semibold text-white tracking-tight">Bank Connections</h1>
          <p className="text-xs text-slate-400">
            Open Banking integrations, consent lifecycles, and ingestion monitoring
          </p>
        </div>
        <Button
          size="sm"
          variant="primary"
          isPending={isConnecting}
          onPress={handleConnectBank}
          className="bg-cyan-600 hover:bg-cyan-500 text-white font-medium text-xs px-3.5 h-8 rounded-xl flex items-center gap-1.5"
        >
          <Plus className="w-3.5 h-3.5" />
          <span>Connect New Bank</span>
        </Button>
      </div>

      {/* Master-Detail Layout */}
      <div className="grid grid-cols-1 lg:grid-cols-12 gap-4 flex-1">
        {/* Left Column: Accounts Roster (4 Cols) */}
        <div className="lg:col-span-4 space-y-3">
          <Card className="glass-card rounded-2xl p-4 border border-slate-800 shadow-none space-y-3">
            <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
              <span className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                Connected Institutions ({accounts.length})
              </span>
              <Chip size="sm" variant="soft" color="success" className="font-mono text-[11px]">
                {accounts.length} Active
              </Chip>
            </div>

            <div className="space-y-2">
              {accounts.length === 0 ? (
                <div className="p-6 text-center text-xs text-slate-500 border border-dashed border-slate-800 rounded-xl">
                  No accounts linked yet. Click above to connect via Open Banking.
                </div>
              ) : (
                accounts.map((acc: any) => {
                  const isSelected = selectedAccount?.account_id === acc.account_id;
                  return (
                    <div
                      key={acc.account_id}
                      onClick={() => setSelectedAccountId(acc.account_id)}
                      className={`cursor-pointer rounded-xl p-3 border transition flex items-center justify-between ${
                        isSelected
                          ? "bg-slate-900/90 border-cyan-500/40 neon-glow-cyan"
                          : "bg-slate-900/50 border-slate-800 hover:border-slate-700"
                      }`}
                    >
                      <div className="flex items-center space-x-3">
                        <div className="w-8 h-8 rounded-lg bg-slate-800 border border-slate-700 flex items-center justify-center font-bold text-white text-xs">
                          {acc.bank_name ? acc.bank_name.charAt(0) : "B"}
                        </div>
                        <div>
                          <p className="text-xs font-medium text-white truncate max-w-[140px]">
                            {acc.bank_name || "Primary Bank"}
                          </p>
                          <p className="text-[11px] font-mono text-slate-400">
                            *{acc.account_number?.slice(-4) || "0000"} • {acc.account_type || "Current"}
                          </p>
                        </div>
                      </div>

                      <div className="text-right">
                        <p className="text-xs font-bold text-white font-mono tabular-nums">
                          £{parseFloat(acc.balance || acc.account_balance || "0").toLocaleString("en-GB", {
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2,
                          })}
                        </p>
                        <span className="inline-flex items-center gap-1 text-[10px] text-emerald-400 font-mono">
                          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400" />
                          Synced
                        </span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>
          </Card>
        </div>

        {/* Right Column: Connection Detail & Sync Management (8 Cols) */}
        <div className="lg:col-span-8 space-y-4">
          {selectedAccount ? (
            <>
              {/* Account Overview Card */}
              <Card className="glass-card rounded-2xl p-5 border border-slate-800 shadow-none space-y-4">
                <div className="flex flex-wrap items-center justify-between gap-3 pb-3 border-b border-slate-800/80">
                  <div className="flex items-center space-x-3">
                    <div className="w-10 h-10 rounded-xl bg-cyan-950/60 border border-cyan-800/60 flex items-center justify-center text-cyan-400">
                      <Building2 className="w-5 h-5" />
                    </div>
                    <div>
                      <h2 className="text-sm font-semibold text-white">
                        {selectedAccount.bank_name || "Institution Name"}
                      </h2>
                      <p className="text-xs font-mono text-slate-400">
                        Account: *{selectedAccount.account_number?.slice(-4) || "0000"} | Sort Code: {selectedAccount.sort_code || "00-00-00"}
                      </p>
                    </div>
                  </div>

                  <div className="text-right">
                    <p className="text-xs text-slate-400">Physical Balance</p>
                    <p className="text-xl font-bold text-white font-mono tabular-nums">
                      £{parseFloat(selectedAccount.balance ? String(selectedAccount.balance) : "0").toLocaleString("en-GB", {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}
                    </p>
                  </div>
                </div>

                {/* Consent & Open Banking Lifecycle */}
                <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 text-xs">
                  <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400 flex items-center gap-1.5">
                        <Calendar className="w-3.5 h-3.5 text-cyan-400" />
                        Consent Validity
                      </span>
                      <Chip size="sm" variant="soft" color="success" className="font-mono text-[10px]">
                        72 Days Remaining
                      </Chip>
                    </div>
                    <ProgressBar aria-label="Consent Validity" value={80} className="w-full">
                      <ProgressBar.Track className="bg-slate-800 h-1.5 rounded-full overflow-hidden">
                        <ProgressBar.Fill className="h-full rounded-full bg-emerald-500" />
                      </ProgressBar.Track>
                    </ProgressBar>
                    <p className="text-[11px] text-slate-500">
                      Standard 90-day FCA Open Banking consent window.
                    </p>
                  </div>

                  <div className="p-3 rounded-xl bg-slate-900/60 border border-slate-800 space-y-2">
                    <div className="flex items-center justify-between">
                      <span className="text-slate-400 flex items-center gap-1.5">
                        <Clock className="w-3.5 h-3.5 text-cyan-400" />
                        Last Webhook Sync
                      </span>
                      <span className="text-white font-mono text-[11px]">8 minutes ago</span>
                    </div>
                    <p className="text-[11px] text-slate-400">
                      Continuous delta sync active. New transactions stream automatically into the priority cascade.
                    </p>
                  </div>
                </div>

                {/* Connection Controls */}
                <div className="flex flex-wrap gap-2 pt-2">
                  <Button
                    size="sm"
                    variant="secondary"
                    isPending={isSyncing}
                    onPress={handleSyncAccount}
                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs px-3 h-8 flex items-center gap-1.5"
                  >
                    <RefreshCcw className="w-3 h-3" />
                    <span>Sync Statement Now</span>
                  </Button>
                  <Button
                    size="sm"
                    variant="secondary"
                    isPending={isExtending}
                    onPress={handleExtendConsent}
                    className="bg-slate-800 hover:bg-slate-700 text-slate-200 border border-slate-700 text-xs px-3 h-8 flex items-center gap-1.5"
                  >
                    <ShieldCheck className="w-3 h-3 text-emerald-400" />
                    <span>Extend 90-Day Consent</span>
                  </Button>
                </div>
              </Card>

              {/* Ingestion Audit Log */}
              <Card className="glass-card rounded-2xl p-5 border border-slate-800 shadow-none space-y-3">
                <div className="flex items-center justify-between pb-2 border-b border-slate-800/80">
                  <h3 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
                    Ingestion Audit Log
                  </h3>
                  <span className="text-[11px] font-mono text-slate-500">Last 24 Hours</span>
                </div>

                <div className="space-y-2 text-xs">
                  <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800/80">
                    <div className="flex items-center space-x-2.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      <div>
                        <p className="font-medium text-white">Delta Ingestion Batch Cleared</p>
                        <p className="text-[11px] text-slate-400 font-mono">3 transactions matched to virtual buckets</p>
                      </div>
                    </div>
                    <span className="text-slate-400 font-mono text-[11px]">Today, 14:20</span>
                  </div>

                  <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800/80">
                    <div className="flex items-center space-x-2.5">
                      <span className="w-2 h-2 rounded-full bg-cyan-400" />
                      <div>
                        <p className="font-medium text-white">Webhook Ping (Zero Delta)</p>
                        <p className="text-[11px] text-slate-400 font-mono">No unbudgeted spend detected</p>
                      </div>
                    </div>
                    <span className="text-slate-400 font-mono text-[11px]">Today, 09:12</span>
                  </div>

                  <div className="flex items-center justify-between p-2.5 rounded-xl bg-slate-900/50 border border-slate-800/80">
                    <div className="flex items-center space-x-2.5">
                      <span className="w-2 h-2 rounded-full bg-emerald-400" />
                      <div>
                        <p className="font-medium text-white">Daily Reconciliation Audit</p>
                        <p className="text-[11px] text-slate-400 font-mono">Zero-sum balance verified: Delta = £0.00</p>
                      </div>
                    </div>
                    <span className="text-slate-400 font-mono text-[11px]">Yesterday, 23:59</span>
                  </div>
                </div>
              </Card>
            </>
          ) : (
            <Card className="glass-card rounded-2xl p-10 border border-slate-800 text-center text-xs text-slate-400">
              Select an account from the roster to view integration details.
            </Card>
          )}
        </div>
      </div>
    </div>
  );
}
