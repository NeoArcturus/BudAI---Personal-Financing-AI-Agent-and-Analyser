"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  RefreshCcw,
  Plus,
  Building2,
  ShieldCheck,
  Lock,
  CheckCircle2,
  AlertCircle,
  ArrowRight,
  CreditCard,
} from "lucide-react";
import { Button, Card, Skeleton, toast } from "@heroui/react";
import { cn } from "@/lib/utils";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import { useTransactions, useConnectionMetadata, ConnectionMetadataItem } from "@/lib/hooks";
import { formatCurrency } from "@/lib/currency";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import Link from "next/link";
import { AnimatedNumber } from "@/app/(protected)/_components/ui/AnimatedNumber";

export default function ConnectionsPage() {
  const queryClient = useQueryClient();
  const router = useRouter();
  const { accounts } = useBudAI();
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [filter, setFilter] = useState<"all" | "action">("all");

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

  const { data: connectionMetadata = [] } = useConnectionMetadata();

  const metadataByBankUuid = useMemo(() => {
    const map = new Map<string, ConnectionMetadataItem>();
    if (connectionMetadata && Array.isArray(connectionMetadata)) {
      for (const item of connectionMetadata) {
        if (item.bank_uuid) {
          map.set(item.bank_uuid, item);
        }
      }
    }
    return map;
  }, [connectionMetadata]);

  const [isReauthenticating, setIsReauthenticating] = useState(false);
  const [isSyncing, setIsSyncing] = useState(false);
  const [failedExtendUuids, setFailedExtendUuids] = useState<Set<string>>(new Set());
  const [revokedUuids, setRevokedUuids] = useState<Set<string>>(new Set());

  const handleReauth = async (bankUuid?: string) => {
    if (!bankUuid) return;
    setIsReauthenticating(true);
    try {
      const cachedMeta = metadataByBankUuid.get(bankUuid);

      // Fast-path: If consent has lapsed or is expiring and reauth_link is already pre-generated
      if (cachedMeta?.reauth_link) {
        window.location.href = cachedMeta.reauth_link;
        return;
      }

      // If metadata flags that re-authentication is required, skip the doomed /extend call
      if (cachedMeta?.needs_reauth) {
        const linkRes = await apiFetch(`/api/auth/banks/${bankUuid}/reauth`, { method: "POST" }, true);
        const linkData = (await linkRes.json()) as any;
        const targetUrl = linkData.auth_uri || linkData.auth_url || linkData.reauth_url;
        if (targetUrl) {
          window.location.href = targetUrl;
          return;
        } else {
          toast("Unable to generate bank re-authentication link.", { variant: "danger" } as any);
          return;
        }
      }

      // Proactive extension attempt for still-valid consent
      const res = await apiFetch(`/api/accounts/banks/${bankUuid}/extend`, { method: "POST" }, true);
      const data = (await res.json()) as any;

      switch (data.status) {
        case "extended":
          toast("90-Day connection access extended successfully.");
          queryClient.invalidateQueries({ queryKey: ["accounts"] });
          queryClient.invalidateQueries({ queryKey: ["connections-metadata"] });
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
          try {
            const linkRes = await apiFetch(`/api/auth/banks/${bankUuid}/reauth`, { method: "POST" }, true);
            const linkData = (await linkRes.json()) as any;
            const targetUrl = linkData.auth_uri || linkData.auth_url || linkData.reauth_url;
            if (targetUrl) {
              window.location.href = targetUrl;
            } else {
              toast("Unable to generate bank re-authentication link.", { variant: "danger" } as any);
            }
          } catch (err: any) {
            toast(err?.message || "Failed to generate re-authentication link.", { variant: "danger" } as any);
          }
          break;
        case "revoked":
          toast("Bank consent was revoked by the institution. Re-linking required.", { variant: "danger" } as any);
          setRevokedUuids((prev) => new Set(prev).add(bankUuid));
          break;
        default:
          break;
      }
    } catch (err: any) {
      toast(err?.message || "Unable to renew connection access.", { variant: "danger" } as any);
    } finally {
      setIsReauthenticating(false);
    }
  };

  const handleConnect = async () => {
    setIsConnecting(true);
    try {
      const res = await apiFetch("/api/auth/truelayer/status", {}, true);
      if (res.ok) {
        const data = (await res.json()) as any;
        if (data.auth_url) {
          router.push(data.auth_url);
        }
      }
    } catch {
      // Error handled by apiFetch
    } finally {
      setIsConnecting(false);
    }
  };

  const handleRefreshAll = async () => {
    setIsSyncing(true);
    try {
      const [metaRes, syncRes] = await Promise.allSettled([
        apiFetch("/api/auth/connections/metadata", {}, true),
        apiFetch("/api/accounts/sync", { method: "POST" }, true),
      ]);

      const metaOk = metaRes.status === "fulfilled" && metaRes.value.ok;
      const syncOk = syncRes.status === "fulfilled" && syncRes.value.ok;

      if (metaOk) {
        queryClient.invalidateQueries({ queryKey: ["connections-metadata"] });
      }
      queryClient.invalidateQueries({ queryKey: ["accounts"] });
      queryClient.invalidateQueries({ queryKey: ["transactions"] });

      if (syncOk && metaOk) {
        toast("Connection status verified. Account transaction and balance sync initiated in background.");
      } else if (syncOk) {
        toast("Account transaction and balance sync initiated in background.");
      } else if (metaOk) {
        toast("Connection access status verified.");
      } else {
        const errorMsg =
          (syncRes.status === "rejected" ? syncRes.reason?.message : null) ||
          (metaRes.status === "rejected" ? metaRes.reason?.message : null) ||
          "Failed to refresh connection status and account data.";
        toast(errorMsg, { variant: "danger" } as any);
      }
    } catch (err: any) {
      toast(err?.message || "Failed to complete account synchronization.", { variant: "danger" } as any);
    } finally {
      setIsSyncing(false);
    }
  };

  const selectedAccount = accounts.find((a) => a.account_id === selectedAccountId);

  // Status definitions
  const STATUS_ACTIVE = "200-200";
  const STATUS_EXPIRED = "200-401";
  const STATUS_REVOKED = "200-403";

  const getAccountHealth = (acc: any) => {
    const meta = acc.bank_uuid ? metadataByBankUuid.get(acc.bank_uuid) : undefined;
    const consentStatus = meta?.consent_status || acc.consent_status;
    const isHardRevoked = consentStatus === STATUS_REVOKED || revokedUuids.has(acc.bank_uuid!);
    const isExpiredStatus =
      consentStatus === STATUS_EXPIRED ||
      isHardRevoked ||
      failedExtendUuids.has(acc.bank_uuid!) ||
      Boolean(meta?.needs_reauth);
    const isActive = consentStatus === STATUS_ACTIVE && !isExpiredStatus;

    if (isHardRevoked) return "red";
    if (isExpiredStatus) return "yellow";
    if (isActive) return "green";
    return "green";
  };

  const getHealthBadge = (health: string) => {
    switch (health) {
      case "green":
        return {
          label: "Active",
          className: "bg-emerald-500/10 text-emerald-600 dark:text-emerald-400 border border-emerald-500/25",
          icon: CheckCircle2,
        };
      case "yellow":
        return {
          label: "Action Required",
          className: "bg-amber-500/10 text-amber-600 dark:text-amber-400 border border-amber-500/25",
          icon: AlertCircle,
        };
      case "red":
      default:
        return {
          label: "Needs Re-link",
          className: "bg-rose-500/10 text-rose-600 dark:text-rose-400 border border-rose-500/25",
          icon: AlertCircle,
        };
    }
  };

  const actionRequiredCount = useMemo(() => {
    return accounts.filter((acc) => getAccountHealth(acc) !== "green").length;
  }, [accounts, revokedUuids, failedExtendUuids, metadataByBankUuid]);

  const filteredAccounts = useMemo(() => {
    if (filter === "action") {
      return accounts.filter((acc) => getAccountHealth(acc) !== "green");
    }
    return accounts;
  }, [accounts, filter, revokedUuids, failedExtendUuids, metadataByBankUuid]);

  // Account specific transactions
  const { data: accountTransactions, isLoading: isTxLoading } = useTransactions(
    selectedAccountId || ""
  );

  const recentTransactions = useMemo(() => {
    if (!accountTransactions || !Array.isArray(accountTransactions)) return [];
    return accountTransactions.slice(0, 5);
  }, [accountTransactions]);

  const selectedHealth = selectedAccount ? getAccountHealth(selectedAccount) : "green";
  const selectedBadge = getHealthBadge(selectedHealth);

  return (
    <div className="w-full h-full overflow-y-auto scrollbar-hide px-4 sm:px-6 lg:px-8 xl:px-12 py-6 pb-28">
      <div className="w-full max-w-[1800px] mx-auto flex flex-col gap-6">

        {/* Page Header */}
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 pb-2 border-b border-border/50">
          <div>
            <h1 className="text-2xl font-bold tracking-tight text-foreground">
              Linked Accounts
            </h1>
            <p className="text-xs text-muted-foreground mt-1">
              Manage your connected bank accounts, review access status, and refresh balances.
            </p>
          </div>
          <div className="flex items-center gap-2">
            <Button
              variant="secondary"
              size="sm"
              onPress={handleRefreshAll}
              isPending={isSyncing}
              className="text-xs font-semibold px-3 py-2 rounded-xl bg-foreground/5 hover:bg-foreground/10 border border-border"
            >
              <RefreshCcw size={14} className={cn("mr-1.5", isSyncing && "animate-spin")} />
              Refresh All
            </Button>
            <Button
              size="sm"
              onPress={handleConnect}
              isPending={isConnecting}
              className="text-xs font-bold px-4 py-2 rounded-xl bg-primary text-primary-foreground hover:opacity-90 shadow-sm"
            >
              <Plus size={14} className="mr-1.5" />
              Link Bank Account
            </Button>
          </div>
        </div>

        {/* Main Content Layout */}
        <div className="flex flex-col lg:flex-row items-start gap-6 w-full">

          {/* Left Column: Accounts List */}
          <div className="w-full lg:w-[340px] xl:w-[380px] shrink-0 flex flex-col gap-4">

            {/* Filter Tabs */}
            <div className="flex items-center justify-between">
              <div className="flex bg-foreground/5 p-1 rounded-xl border border-border gap-1 w-full">
                <button
                  type="button"
                  onClick={() => setFilter("all")}
                  className={cn(
                    "flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold transition-colors text-center cursor-pointer",
                    filter === "all"
                      ? "bg-popover text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  All ({accounts.length})
                </button>
                <button
                  type="button"
                  onClick={() => setFilter("action")}
                  className={cn(
                    "flex-1 py-1.5 px-3 rounded-lg text-xs font-semibold transition-colors text-center cursor-pointer",
                    filter === "action"
                      ? "bg-popover text-foreground shadow-sm"
                      : "text-muted-foreground hover:text-foreground"
                  )}
                >
                  Action Required ({actionRequiredCount})
                </button>
              </div>
            </div>

            {/* Accounts Cards List */}
            <div className="flex flex-col gap-3">
              {isLoading ? (
                Array(3)
                  .fill(0)
                  .map((_, i) => (
                    <div
                      key={i}
                      className="p-4 rounded-2xl border border-border bg-popover shadow-sm flex flex-col gap-2"
                    >
                      <Skeleton className="w-2/3 h-4 rounded-md" />
                      <Skeleton className="w-1/2 h-6 rounded-md" />
                      <Skeleton className="w-1/3 h-3 rounded-md" />
                    </div>
                  ))
              ) : filteredAccounts.length === 0 ? (
                <div className="p-8 rounded-2xl border border-dashed border-border text-center text-muted-foreground text-xs">
                  No accounts matching your filter.
                </div>
              ) : (
                filteredAccounts.map((acc: any) => {
                  const isSelected = selectedAccountId === acc.account_id;
                  const health = getAccountHealth(acc);
                  const badge = getHealthBadge(health);
                  const BadgeIcon = badge.icon;

                  return (
                    <div
                      key={acc.account_id}
                      onClick={() => setSelectedAccountId(acc.account_id)}
                      className={cn(
                        "cursor-pointer rounded-2xl transition-all p-4 text-left border shadow-sm flex flex-col gap-3 relative overflow-hidden",
                        isSelected
                          ? "border-primary/50 bg-primary/5 shadow-[0_0_20px_rgba(0,242,255,0.06)]"
                          : "border-border bg-popover hover:border-foreground/20"
                      )}
                    >
                      <div className="flex items-center justify-between gap-2">
                        <div className="flex items-center gap-2.5 truncate">
                          <div className="w-8 h-8 rounded-xl bg-foreground/5 border border-border flex items-center justify-center shrink-0 text-foreground">
                            <Building2 size={15} />
                          </div>
                          <div className="truncate">
                            <h3 className="font-bold text-xs text-foreground truncate">
                              {acc.bank_name || "Bank Account"}
                            </h3>
                            <p className="text-[11px] text-muted-foreground truncate">
                              Current Account
                            </p>
                          </div>
                        </div>

                        <span
                          className={cn(
                            "inline-flex items-center gap-1 px-2 py-0.5 rounded-full text-[10px] font-semibold shrink-0",
                            badge.className
                          )}
                        >
                          <BadgeIcon size={11} />
                          {badge.label}
                        </span>
                      </div>

                      <div className="flex items-baseline justify-between pt-1 border-t border-border/40">
                        <span className="font-mono text-sm font-bold text-foreground">
                          {formatCurrency(acc.balance || 0, acc.currency || "GBP")}
                        </span>
                        <span className="font-mono text-[11px] text-muted-foreground">
                          *{acc.account_number?.slice(-4) || "0000"}
                        </span>
                      </div>
                    </div>
                  );
                })
              )}
            </div>

            {/* Pinned Link Account Card */}
            <div className="p-4 rounded-2xl border border-dashed border-border bg-foreground/[0.02] flex flex-col gap-2 text-center items-center">
              <p className="text-xs font-semibold text-foreground">
                Connect Another Bank
              </p>
              <p className="text-[11px] text-muted-foreground">
                Sync additional current or savings accounts securely.
              </p>
              <Button
                size="sm"
                variant="secondary"
                onPress={handleConnect}
                isPending={isConnecting}
                className="mt-1 w-full text-xs font-semibold rounded-xl bg-foreground/5 hover:bg-foreground/10 border border-border"
              >
                <Plus size={13} className="mr-1" />
                Add Bank
              </Button>
            </div>
          </div>

          {/* Right Column: Selected Account Overview & Activity */}
          <div className="w-full lg:flex-1 min-w-0 flex flex-col gap-6">
            {!selectedAccount ? (
              <div className="p-12 rounded-2xl border border-dashed border-border text-center text-muted-foreground text-xs">
                Select an account from the list to view details and activity.
              </div>
            ) : (
              <div className="flex flex-col gap-6 w-full">

                {/* Primary Account Overview Card */}
                <div className="p-6 rounded-2xl border border-border bg-popover shadow-xl flex flex-col md:flex-row justify-between md:items-center gap-6">
                  <div className="flex items-start gap-4">
                    <div className="w-12 h-12 rounded-2xl bg-foreground/5 border border-border flex items-center justify-center shrink-0 text-foreground">
                      <Building2 size={24} />
                    </div>
                    <div>
                      <div className="flex items-center gap-2">
                        <h2 className="text-lg font-bold text-foreground">
                          {selectedAccount.bank_name || "Bank Account"}
                        </h2>
                        <span
                          className={cn(
                            "inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[10px] font-semibold",
                            selectedBadge.className
                          )}
                        >
                          <selectedBadge.icon size={11} />
                          {selectedBadge.label}
                        </span>
                      </div>
                      <p className="text-xs font-mono text-muted-foreground mt-0.5">
                        Account: **** {selectedAccount.account_number?.slice(-4) || "0000"} • Sort Code: {selectedAccount.sort_code || "00-00-00"}
                      </p>
                    </div>
                  </div>

                  {/* Balance & Quick Actions */}
                  <div className="flex flex-col md:items-end gap-3 shrink-0">
                    <div>
                      <div className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold md:text-right">
                        Available Balance
                      </div>
                      <div className="text-2xl font-bold font-mono text-foreground mt-0.5 md:text-right">
                        <AnimatedNumber
                          value={selectedAccount.balance || 0}
                          currency={selectedAccount.currency || "GBP"}
                        />
                      </div>
                    </div>

                    <div className="flex items-center gap-2">
                      <Button
                        size="sm"
                        variant="secondary"
                        onPress={handleRefreshAll}
                        isPending={isSyncing}
                        className="text-xs font-semibold px-3 py-1.5 rounded-xl bg-foreground/5 hover:bg-foreground/10 border border-border"
                      >
                        <RefreshCcw size={13} className={cn("mr-1.5", isSyncing && "animate-spin")} />
                        Refresh
                      </Button>
                      <Button
                        size="sm"
                        onPress={() => handleReauth(selectedAccount.bank_uuid)}
                        isPending={isReauthenticating}
                        className="text-xs font-bold px-3.5 py-1.5 rounded-xl bg-primary text-primary-foreground hover:opacity-90 shadow-sm"
                      >
                        <ShieldCheck size={14} className="mr-1.5" />
                        Renew 90-Day Access
                      </Button>
                    </div>
                  </div>
                </div>

                {/* Details & Governance Grid */}
                <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">

                  {/* Account Specifications */}
                  <div className="p-6 rounded-2xl border border-border bg-popover shadow-sm flex flex-col gap-4">
                    <div className="flex items-center gap-2 pb-2 border-b border-border/50">
                      <CreditCard size={15} className="text-primary" />
                      <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">
                        Account Details
                      </h3>
                    </div>

                    <div className="grid grid-cols-2 gap-y-3.5 text-xs">
                      <div>
                        <span className="text-muted-foreground block text-[11px]">Institution</span>
                        <span className="font-semibold text-foreground mt-0.5 block truncate">
                          {selectedAccount.bank_name || "Registered Bank"}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground block text-[11px]">Account Type</span>
                        <span className="font-semibold text-foreground mt-0.5 block">
                          Current Account
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground block text-[11px]">Account Number</span>
                        <span className="font-mono font-semibold text-foreground mt-0.5 block">
                          **** {selectedAccount.account_number?.slice(-4) || "0000"}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground block text-[11px]">Sort Code</span>
                        <span className="font-mono font-semibold text-foreground mt-0.5 block">
                          {selectedAccount.sort_code || "00-00-00"}
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground block text-[11px]">Currency</span>
                        <span className="font-semibold text-foreground mt-0.5 block">
                          {selectedAccount.currency || "GBP"} (£)
                        </span>
                      </div>
                      <div>
                        <span className="text-muted-foreground block text-[11px]">Connection Type</span>
                        <span className="font-semibold text-foreground mt-0.5 block">
                          UK Open Banking
                        </span>
                      </div>
                    </div>
                  </div>

                  {/* Security & Access Protection */}
                  <div className="p-6 rounded-2xl border border-border bg-popover shadow-sm flex flex-col gap-4">
                    <div className="flex items-center gap-2 pb-2 border-b border-border/50">
                      <Lock size={15} className="text-primary" />
                      <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">
                        Security & Data Privacy
                      </h3>
                    </div>

                    <div className="flex flex-col gap-3 text-xs">
                      <div className="flex items-start gap-2.5">
                        <CheckCircle2 size={14} className="text-emerald-500 shrink-0 mt-0.5" />
                        <div>
                          <span className="font-semibold text-foreground block">
                            Read-Only Access
                          </span>
                          <span className="text-[11px] text-muted-foreground leading-relaxed block mt-0.5">
                            BudAI can only read account balances and transactions. It cannot initiate real-world payments or transfer money out of your bank.
                          </span>
                        </div>
                      </div>

                      <div className="flex items-start gap-2.5">
                        <CheckCircle2 size={14} className="text-emerald-500 shrink-0 mt-0.5" />
                        <div>
                          <span className="font-semibold text-foreground block">
                            256-bit TLS Encryption
                          </span>
                          <span className="text-[11px] text-muted-foreground leading-relaxed block mt-0.5">
                            All communication with bank APIs is securely encrypted via FCA-regulated Open Banking protocols.
                          </span>
                        </div>
                      </div>

                      <div className="flex items-start gap-2.5">
                        <CheckCircle2 size={14} className="text-emerald-500 shrink-0 mt-0.5" />
                        <div>
                          <span className="font-semibold text-foreground block">
                            90-Day Renewal Standard
                          </span>
                          <span className="text-[11px] text-muted-foreground leading-relaxed block mt-0.5">
                            UK Open Banking standards require confirming your consent every 90 days to ensure you maintain total control.
                          </span>
                        </div>
                      </div>
                    </div>
                  </div>
                </div>

                {/* Recent Account Activity Card */}
                <div className="p-6 rounded-2xl border border-border bg-popover shadow-sm flex flex-col gap-4">
                  <div className="flex items-center justify-between pb-2 border-b border-border/50">
                    <div>
                      <h3 className="text-xs font-bold uppercase tracking-wider text-foreground">
                        Recent Account Activity
                      </h3>
                      <p className="text-[11px] text-muted-foreground mt-0.5">
                        Latest transactions recorded for this account
                      </p>
                    </div>

                    <Link
                      href={`/transactions`}
                      className="text-xs font-semibold text-primary hover:underline flex items-center gap-1"
                    >
                      <span>View all</span>
                      <ArrowRight size={13} />
                    </Link>
                  </div>

                  <div className="flex flex-col divide-y divide-border/40">
                    {isTxLoading ? (
                      Array(3)
                        .fill(0)
                        .map((_, i) => (
                          <div key={i} className="py-3 flex items-center justify-between">
                            <Skeleton className="w-1/3 h-4 rounded-md" />
                            <Skeleton className="w-16 h-4 rounded-md" />
                          </div>
                        ))
                    ) : recentTransactions.length === 0 ? (
                      <div className="py-6 text-center text-xs text-muted-foreground">
                        No recent transactions recorded for this account.
                      </div>
                    ) : (
                      recentTransactions.map((tx: any, idx: number) => {
                        const amt = tx.amount || tx.Amount || 0;
                        const desc = tx.description || tx.merchant_name || "Transaction";
                        const cat = tx.category || tx.Category || "General";
                        const dateStr = tx.date || tx.timestamp;
                        const formattedDate = dateStr
                          ? new Date(dateStr).toLocaleDateString("en-GB", {
                              day: "numeric",
                              month: "short",
                              year: "numeric",
                            })
                          : "";

                        return (
                          <div
                            key={tx.transaction_id || tx.transaction_uuid || idx}
                            className="py-3 flex items-center justify-between gap-4"
                          >
                            <div className="flex flex-col truncate">
                              <span className="font-semibold text-xs text-foreground truncate">
                                {desc}
                              </span>
                              <span className="text-[11px] text-muted-foreground mt-0.5">
                                {formattedDate} • {cat}
                              </span>
                            </div>

                            <span
                              className={cn(
                                "font-mono font-bold text-xs shrink-0",
                                amt > 0 ? "text-emerald-500" : "text-foreground"
                              )}
                            >
                              {formatCurrency(amt, selectedAccount.currency || "GBP")}
                            </span>
                          </div>
                        );
                      })
                    )}
                  </div>
                </div>

              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
