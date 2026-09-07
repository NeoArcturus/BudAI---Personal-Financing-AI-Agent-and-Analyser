"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  RefreshCcw,
  Plus,
  Activity,
  Database,
  AlertTriangle,
  Search,
  CheckCircle2,
  XCircle
} from "lucide-react";
import { Button, Card, Skeleton, toast, } from "@heroui/react";
import { cn } from "@/lib/utils";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";








import { AnimatedNumber } from "@/app/(protected)/_components/ui/AnimatedNumber";
import {
  ConnectionHealthWidget,
  ConnectionExpenseWidget,
  ConnectionHabitsWidget,
  ConnectionBucketWidget
} from "./ConnectionWidgets";
import { Target, Repeat, CreditCard } from "lucide-react";


export default function ConnectionsPage() {
  const queryClient = useQueryClient();
  const router = useRouter();
  const { accounts } = useBudAI();
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState("all");

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);

    if (accounts.length > 0) {
      if (!selectedAccountId) {
        setSelectedAccountId(accounts[0].account_id);
      }
      setIsLoading(false);
      clearTimeout(timer);
    }
    return () => clearTimeout(timer);
  }, [accounts, selectedAccountId]);

  const [isReauthenticating, setIsReauthenticating] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [failedExtendUuids, setFailedExtendUuids] = useState<Set<string>>(new Set());
  const [revokedUuids, setRevokedUuids] = useState<Set<string>>(new Set());

  const handleReauth = async (bankUuid?: string) => {
    if (!bankUuid) return;
    setIsReauthenticating(true);
    try {
      const res = await apiFetch(`/api/accounts/banks/${bankUuid}/extend`, { method: "POST" }, true);
      const data = await res.json() as any;

      switch (data.status) {
        case "extended":
          toast.success("Connection securely extended for 90 days!");
          queryClient.invalidateQueries({ queryKey: ["accounts"] });
          setFailedExtendUuids((prev) => {
            const next = new Set(prev);
            next.delete(bankUuid);
            return next;
          });
          setRevokedUuids((prev) => {
            const next = new Set(prev);
            next.delete(bankUuid);
            return next;
          });
          break;
        case "reauth_required":
          toast.warning("Manual re-authentication required. Redirecting...");
          try {
            const linkRes = await apiFetch(`/api/auth/banks/${bankUuid}/reauth`, { method: "POST" }, true);
            const linkData = await linkRes.json() as any;
            if (linkData.auth_uri || linkData.auth_url || linkData.reauth_url) {
              window.location.href = linkData.auth_uri || linkData.auth_url || linkData.reauth_url;
            }
          } catch (err) {
            toast.danger("Failed to generate secure re-auth link.");
          }
          break;
        case "revoked":
          toast.danger("Bank revoked access. Please re-link.");
          setRevokedUuids((prev) => new Set(prev).add(bankUuid));
          break;
        default:
          toast.danger("Unable to reach the bank. Please try again.");
          break;
      }
    } catch (error) {
      toast.danger("Unable to reach the bank.");
    } finally {
      setIsReauthenticating(false);
    }
  };

  const handleConnect = async () => {
    setIsConnecting(true);
    try {
      const res = await apiFetch("/api/auth/truelayer/status", {}, true);
      if (res.ok) {
        const data = await res.json() as any;
        if (data.auth_url) {
          router.push(data.auth_url);
        }
      }
    } catch (error) {
      toast.danger("Failed to initiate bank connection");
    } finally {
      setIsConnecting(false);
    }
  };

  const handleRefreshAll = async () => {
    setIsSyncing(true);
    try {
      toast.success("Syncing all accounts...");
      // Simulate sync delay for UX
      await new Promise(r => setTimeout(r, 1500));
      queryClient.invalidateQueries({ queryKey: ["accounts"] });
      toast.success("All connections updated.");
    } catch (e) {
      toast.danger("Failed to sync all accounts.");
    } finally {
      setIsSyncing(false);
    }
  };

  const selectedAccount = accounts.find((a) => a.account_id === selectedAccountId);

  // Determine health status helper
  const getAccountHealth = (acc: any) => {
    const isHardRevoked = acc.consent_status === "200-403" || revokedUuids.has(acc.bank_uuid!);
    const isExpiredStatus = acc.consent_status === "200-401" || isHardRevoked || failedExtendUuids.has(acc.bank_uuid!);

    if (isHardRevoked) return "red";
    if (isExpiredStatus) return "yellow";
    return "green";
  };

  const filteredAccounts = useMemo(() => {
    if (filter === "action") {
      return accounts.filter(acc => getAccountHealth(acc) !== "green");
    }
    return accounts;
  }, [accounts, filter, revokedUuids, failedExtendUuids]);

  return (
    <div className="w-full flex-1 flex flex-col h-full bg-transparent text-foreground font-sans relative overflow-hidden transition-colors duration-500">

      {/* Dynamic Grid Background overlay */}
      <div className="pointer-events-none absolute inset-0 z-0 opacity-20 transition-opacity duration-700">
        <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_0%,rgba(0,242,255,0.03),transparent_70%)]" />
        <div className="absolute inset-0 bg-[linear-gradient(to_right,rgba(255,255,255,0.02)_1px,transparent_1px),linear-gradient(to_bottom,rgba(255,255,255,0.02)_1px,transparent_1px)] bg-[size:4rem_4rem] [mask-image:radial-gradient(ellipse_60%_60%_at_50%_0%,#000_70%,transparent_100%)]" />
      </div>

      <div className="relative z-10 flex-1 flex flex-col pt-10 px-10 h-full">
        <div className="flex items-center justify-between mb-10 shrink-0">
          <div>
            <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic">
              Data <span className="font-normal not-italic">Connections</span>
            </h2>
            <p className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.4em] mt-1.5">
              Live Bank Ingestion Matrix
            </p>
          </div>
        </div>

        <div className="flex-1 flex flex-col lg:flex-row gap-8 pb-8 overflow-hidden h-full">

          {/* Left Panel: The Connection Roster (Control Center Sidebar) */}
          <div className="w-full lg:w-80 shrink-0 flex flex-col gap-4 overflow-y-auto custom-scrollbar relative h-full">

            <div className="flex items-center justify-between px-2 shrink-0">
              <div className="flex bg-white/5 rounded-lg p-1 gap-1">
                <button
                  onClick={() => setFilter("all")}
                  className={cn("px-4 py-1.5 rounded-md text-[9px] font-black uppercase tracking-widest transition-colors", filter === "all" ? "bg-white/10 text-white" : "text-foreground/40 hover:text-foreground/70")}
                >
                  All
                </button>
                <button
                  onClick={() => setFilter("action")}
                  className={cn("px-4 py-1.5 rounded-md text-[9px] font-black uppercase tracking-widest transition-colors", filter === "action" ? "bg-white/10 text-white" : "text-foreground/40 hover:text-foreground/70")}
                >
                  Action Needed
                </button>
              </div>

              <Button
                isIconOnly
                onPress={handleRefreshAll}
                isPending={isSyncing}
                className="bg-transparent hover:bg-white/5 text-foreground/40 hover:text-foreground transition-all rounded-lg"
              >
                <RefreshCcw size={14} className={cn(isSyncing && "animate-spin")} />
              </Button>
            </div>

            <div className="flex-1 overflow-y-auto space-y-3 pb-24">
              {isLoading ? (
                Array(3).fill(0).map((_, i) => (
                  <Card key={i} className="rounded-xl p-5 shadow-inner h-20 flex items-center justify-center">
                    <Skeleton className="w-full h-8 bg-white/5 rounded-lg" />
                  </Card>
                ))
              ) : filteredAccounts.length === 0 ? (
                <Card className="rounded-xl p-8 shadow-inner text-center text-foreground/40 text-[10px] font-mono uppercase tracking-widest border-white/5 border border-dashed bg-transparent mt-4">
                  No Accounts Found
                </Card>
              ) : (
                filteredAccounts.map((acc: any) => {
                  const health = getAccountHealth(acc);
                  const isSelected = selectedAccountId === acc.account_id;

                  return (
                    <div
                      key={acc.account_id}
                      onClick={() => setSelectedAccountId(acc.account_id)}
                      className={cn(
                        "cursor-pointer rounded-2xl transition-all p-5 text-left flex items-center gap-4 w-full border relative overflow-hidden group hover:bg-white/5",
                        isSelected ? "border-primary/50 bg-primary/5 shadow-[0_0_20px_rgba(0,242,255,0.05)]" : "border-white/5 bg-black/40"
                      )}
                    >
                      {/* Traffic Light Dot */}
                      <div className="flex shrink-0 items-center justify-center">
                        <div className={cn(
                          "w-2 h-2 rounded-full",
                          health === "red" ? "bg-red-500 shadow-[0_0_8px_rgba(239,68,68,0.5)]" :
                            health === "yellow" ? "bg-yellow-500 shadow-[0_0_8px_rgba(234,179,8,0.5)]" :
                              "bg-green-500 shadow-[0_0_8px_rgba(34,197,94,0.5)]"
                        )} />
                      </div>

                      <div className="flex-1 overflow-hidden">
                        <h4 className="font-black text-xs tracking-widest uppercase truncate text-foreground group-hover:text-primary transition-colors">
                          {acc.bank_name || "Unknown Bank"}
                        </h4>
                        <p className="text-[9px] font-mono tracking-[0.2em] mt-1 uppercase text-foreground/40">
                          *{acc.account_number?.slice(-4) || "0000"} • Updated 10m ago
                        </p>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {/* Pinned Add Bank Button */}
            <div className="absolute bottom-0 left-0 right-0 p-4">
              <Button
                onPress={handleConnect}
                isPending={isConnecting}
                className="w-full flex items-center justify-center gap-3 font-black text-[10px] uppercase tracking-widest rounded-xl h-12 transition-all cursor-pointer bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 shadow-[0_0_15px_rgba(0,242,255,0.1)] hover:shadow-[0_0_20px_rgba(0,242,255,0.2)]"
              >
                <Plus size={16} /> Connect Bank
              </Button>
            </div>
          </div>

          {/* Right Panel: Detailed Profile & Embedded Dashboard */}
          <div className="flex-1 flex flex-col gap-6 overflow-y-auto custom-scrollbar h-full pb-10 pr-2">

            {!selectedAccount ? (
              <div className="flex items-center justify-center h-64 border border-dashed border-white/10 rounded-2xl">
                <span className="text-foreground/30 font-mono text-[10px] uppercase tracking-widest">Select an account</span>
              </div>
            ) : (
              <div className="flex flex-col gap-6 w-full">

                {/* Header Card (Kept simple to identify selected node) */}
                <Card className="rounded-2xl p-8 flex flex-col md:flex-row justify-between md:items-end gap-6 overflow-hidden relative border border-white/5 bg-content1">

                  {/* Glass Background Decor */}
                  <div className="absolute right-0 top-0 w-64 h-64 bg-primary/5 rounded-full blur-3xl -translate-y-1/2 translate-x-1/3 pointer-events-none" />

                  <div className="relative z-10">
                    <p className="text-[9px] font-black text-primary uppercase tracking-[0.3em] mb-2 flex items-center gap-2">
                      <Database size={12} /> Active Connection
                    </p>
                    <h2 className="text-3xl font-black text-foreground uppercase tracking-tight italic mb-1 truncate">
                      {selectedAccount.bank_name}
                    </h2>
                    <p className="text-xs font-mono text-foreground/40 uppercase tracking-widest">
                      {selectedAccount.account_number || "**** 0000"} | {selectedAccount.sort_code || "00-00-00"}
                    </p>
                  </div>

                  <div className="relative z-10 text-left md:text-right">
                    <p className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.3em] mb-1">
                      Available Balance
                    </p>
                    <div className="text-4xl font-mono text-foreground font-black tracking-tighter">
                      {selectedAccount.balance ? <div className="flex items-center gap-1 justify-end"><AnimatedNumber value={selectedAccount.balance} currency={selectedAccount.currency} /></div> : <Skeleton className="h-8 w-32 bg-white/5 rounded-lg ml-auto" />}
                    </div>
                  </div>
                </Card>

                <div className="grid grid-cols-1 xl:grid-cols-2 gap-6 w-full auto-rows-[400px]">
                  <ConnectionHealthWidget accountId={selectedAccountId || ""} />
                  <ConnectionExpenseWidget accountId={selectedAccountId || ""} />
                  <ConnectionHabitsWidget accountId={selectedAccountId || ""} />
                  <ConnectionBucketWidget accountId={selectedAccountId || ""} type="RECURRING" title="Recurring Payments" icon={Repeat} />
                  <ConnectionBucketWidget accountId={selectedAccountId || ""} type="TARGET_DATE" title="Goals Progress" icon={Target} />
                  <ConnectionBucketWidget accountId={selectedAccountId || ""} type="LIABILITY" title="Liabilities" icon={CreditCard} />
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
