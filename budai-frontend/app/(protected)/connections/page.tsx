"use client";

import React, { useState, useEffect, useMemo } from "react";
import {
  RefreshCcw,
  CreditCard,
  Plus,
  Activity,
  Database,
  PieChart,
  ListOrdered,
  Repeat
} from "lucide-react";
import { Button, Card, Skeleton, toast, Table, Switch } from "@heroui/react";
import { cn } from "@/lib/utils";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch, getUserUuid } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import { useTransactions, useExpenseCategories } from "@/lib/hooks";
import { useSubscriptionsData } from "@/app/(protected)/_hooks/useSubscriptionsData";
import CoreChartEngine from "@/app/(protected)/_components/internal/ChartEngine";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { today, getLocalTimeZone } from "@internationalized/date";

export default function ConnectionsPage() {
  const queryClient = useQueryClient();
  const router = useRouter();
  const { accounts } = useBudAI();
  const [selectedAccountId, setSelectedAccountId] = useState<string | null>(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

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
  const [isSurvivalMode, setIsSurvivalMode] = useState<boolean>(false);

  const handleReauth = async (bankUuid?: string) => {
    if (!bankUuid) return;
    setIsReauthenticating(true);
    try {
      const res = await apiFetch(`/api/accounts/banks/${bankUuid}/extend`, { method: "POST" }, true);
      if (res.ok) {
        const data = await res.json() as any;
        if (data.status === "extended") {
          toast.success("Connection securely extended for 90 days!");
          queryClient.invalidateQueries({ queryKey: ["accounts"] });
        } else if (data.status === "reauth_required") {
          toast.warning("Manual re-authentication required. Redirecting securely...");
          if (data.auth_uri) {
            window.location.href = data.auth_uri;
          }
        } else if (data.status === "revoked") {
          toast.danger("Your bank has revoked access. Please re-link this account from scratch.");
        }
      } else {
        toast.danger("Failed to initialize secure connection");
      }
    } catch (error) {
      toast.danger("Failed to initialize secure connection");
    } finally {
      setIsReauthenticating(false);
    }
  };

  const [isSyncing, setIsSyncing] = useState(false);

  const handleDeepSync = async (bankUuid?: string) => {
    if (!bankUuid) return;
    setIsSyncing(true);
    try {
      const res = await apiFetch(`/api/banks/${bankUuid}/sync`, { method: "POST" }, true);
      if (res.ok) {
        toast.success("Deep sync initialized!");
      } else {
        toast.danger("Failed to sync account");
      }
    } catch (error) {
      toast.danger("Failed to sync account");
    } finally {
      setIsSyncing(false);
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
      console.error(error);
      toast.danger("Failed to initiate bank connection");
    } finally {
      setIsConnecting(false);
    }
  };

  const selectedAccount = accounts.find((a) => a.account_id === selectedAccountId);

  // --- Data Fetching for Selected Account ---
  const endDate = today(getLocalTimeZone());
  const startDate = endDate.subtract({ months: 1 });
  const fromStr = `${startDate.year}-${String(startDate.month).padStart(2, "0")}-${String(startDate.day).padStart(2, "0")}`;
  const toStr = `${endDate.year}-${String(endDate.month).padStart(2, "0")}-${String(endDate.day).padStart(2, "0")}`;

  const { data: transactions = [], isFetching: isTxFetching } = useTransactions(
    selectedAccount?.account_id || "",
    fromStr,
    toStr
  );

  const { data: expenseData = [], isFetching: isExpenseFetching } = useExpenseCategories(
    selectedAccount?.account_id || "",
    fromStr,
    toStr,
    true
  );

  const userUuid = typeof window !== "undefined" ? getUserUuid() : null;
  const { subscriptionsData, isLoading: isSubFetching } = useSubscriptionsData(userUuid);

  // Filter subscriptions to only those matching this account's sort_code and account_number if they exist
  // We do a loose match or just bank_name match since TrueLayer subscriptions may not perfectly map to account_ids
  const accountSubscriptions = useMemo(() => {
    if (!subscriptionsData?.subscriptions) return [];
    if (!selectedAccount) return [];
    return subscriptionsData.subscriptions.filter((sub: any) => {
      // Strict match by account_id now that the backend provides it
      if (sub.account_id && selectedAccount.account_id) {
        return sub.account_id === selectedAccount.account_id;
      }

      // Legacy fallback just in case
      if (sub.account_number && selectedAccount.account_number) {
        return sub.account_number === selectedAccount.account_number && sub.sort_code === selectedAccount.sort_code;
      }
      if (sub.bank_name && selectedAccount.bank_name) {
        return sub.bank_name.toLowerCase() === selectedAccount.bank_name.toLowerCase();
      }
      return true;
    }).sort((a: any, b: any) => {
      if (a.status === '303-410' && b.status !== '303-410') return 1;
      if (a.status !== '303-410' && b.status === '303-410') return -1;
      return 0;
    });
  }, [subscriptionsData, selectedAccount]);

  const pieChartConfig = useMemo(() => {
    if (!expenseData || expenseData.length === 0 || !expenseData[0].data) return null;

    const aggregatedData = expenseData[0].data
      .map((item: any) => ({
        name: item.Category || item.category || item.Date || item.date || "Other",
        value: item.Total_Amount || item.total_amount || item.Amount || item.amount || 0,
      }))
      .filter((item) => item.value > 0)
      .sort((a, b) => b.value - a.value);

    if (aggregatedData.length === 0) return null;

    const payload = [
      {
        bank_name: selectedAccount?.bank_name || "Account",
        data: aggregatedData.map((d) => ({
          Category: d.name,
          Total_Amount: d.value,
        })),
      },
    ];

    return buildChartConfig("categorized_doughnut", payload, {
      bank_name_or_id: selectedAccount?.account_id,
      from_date: fromStr,
      to_date: toStr,
    }, "Expenses", { disableAnimation: true });
  }, [expenseData, selectedAccount, fromStr, toStr]);

  const getCategoryTheme = (category: string) => {
    const themes: Record<string, string> = {
      "Food & Dining": "bg-pink-500/20 text-pink-500 border-pink-500/40",
      Groceries: "bg-orange-500/20 text-orange-500 border-orange-500/40",
      Transportation: "bg-yellow-500/20 text-yellow-500 border-yellow-500/40",
      "Bills & Utilities": "bg-cyan-300/20 text-cyan-400 border-cyan-300/40",
      Rent: "bg-blue-500/20 text-blue-500 border-blue-500/40",
      Shopping: "bg-purple-500/20 text-purple-500 border-purple-500/40",
      Entertainment: "bg-destructive/20 text-destructive border-destructive/40",
      "Health & Wellness": "bg-green-500/20 text-green-500 border-green-500/40",
      "Transfers & Investments": "bg-emerald-500/20 text-emerald-500 border-emerald-500/40",
      "High-Risk / Anomaly": "bg-red-500/20 text-red-500 border-red-500/40",
      Salary: "bg-green-500/20 text-green-500 border-green-500/40",
      Income: "bg-green-500/20 text-green-500 border-green-500/40",
      Travel: "bg-yellow-500/20 text-yellow-500 border-yellow-500/40",
      Education: "bg-lime-500/20 text-lime-500 border-lime-500/40",
    };
    return themes[category] || "bg-muted text-muted-foreground border-muted-foreground/40";
  };

  const recentTransactions = React.useMemo(() => {
    let list = Array.isArray(transactions) ? transactions : [];
    if (isSurvivalMode) {
      list = list.filter((tx) => {
        const tags = tx.tags || [];
        return tags.includes("#essential") || tags.includes("#housing") || tags.includes("#recurring");
      });
    }
    return list.slice(0, 20);
  }, [transactions, isSurvivalMode]);

  const isHardRevoked = selectedAccount ? selectedAccount.consent_status === "200-403" : false;

  return (
    <div className="relative z-10 flex-1 flex flex-col pt-10 px-10 h-full">
      <div className="flex items-center justify-between mb-10 shrink-0">
        <div>
          <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic">
            Data <span className="font-normal not-italic">Connections</span>
          </h2>
          <p className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.4em] mt-1.5">
            Institution Management
          </p>
        </div>
        <Button
          onPress={handleConnect}
          isPending={isConnecting}
          className="flex items-center justify-center gap-3 font-black text-[10px] uppercase tracking-widest rounded-xl px-8 h-12 hover:scale-[1.02] transition-all cursor-pointer bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 shadow-lg"
        >
          <Plus size={16} /> Connect Bank
        </Button>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-12 gap-8 pb-8 overflow-hidden h-full">
        {/* Left Sidebar: Accounts List */}
        <div className="lg:col-span-4 flex flex-col gap-4 overflow-y-auto scrollbar-hide">
          <h3 className="text-[10px] font-black text-foreground/30 uppercase tracking-[0.4em] px-2 italic shrink-0">
            Connected Bank accounts
          </h3>
          {isLoading ? (
            Array.from({ length: 3 }).map((_, i) => (
              <Card key={i} className="liquid-glass rounded-xl p-6 border-none shadow-inner h-24 flex items-center justify-center">
                <Skeleton animationType="shimmer" className="w-full h-8 bg-white/5 rounded-lg" />
              </Card>
            ))
          ) : accounts.length === 0 && !isConnecting ? (
            <Card className="liquid-glass rounded-xl p-8 border-none shadow-inner text-center text-foreground/40 text-xs font-mono uppercase">
              No Accounts Linked
            </Card>
          ) : (
            accounts.map((acc) => {
              const isExpired = acc.consent_status === "200-401" || acc.consent_status === "200-403";
              return (
                <Card
                  key={acc.account_id}
                  onClick={() => setSelectedAccountId(acc.account_id)}
                  className={`cursor-pointer liquid-glass rounded-xl transition-all group p-6 shadow-inner text-left flex flex-col items-start w-full border-[1px] relative overflow-hidden ${isExpired
                    ? selectedAccountId === acc.account_id ? "border-red-500/50 bg-red-500/10 grayscale-0 opacity-100" : "border-red-500/20 grayscale opacity-60 hover:opacity-100"
                    : selectedAccountId === acc.account_id ? "border-primary/50 bg-primary/5" : "border-transparent hover:border-white/10"
                    }`}
                >
                  {isExpired && (
                    <div className="absolute right-0 top-0 w-8 h-8 bg-red-500/20 rounded-bl-xl flex justify-center items-center">
                      <span className="text-xs">⚠️</span>
                    </div>
                  )}
                  <div className="flex items-center gap-4 w-full">
                    <div className={`w-10 h-10 rounded-lg flex items-center justify-center shrink-0 transition-colors ${isExpired ? "bg-red-500/20 text-red-500" :
                      selectedAccountId === acc.account_id ? "bg-primary/20 text-primary" : "bg-white/5 text-foreground/50 group-hover:text-foreground"
                      }`}>
                      <CreditCard size={18} />
                    </div>
                    <div className="flex-1 overflow-hidden">
                      <h4 className={`font-black text-sm tracking-tighter uppercase italic truncate ${isExpired ? "text-red-500" :
                        selectedAccountId === acc.account_id ? "text-primary" : "text-foreground"
                        }`}>
                        {acc.bank_name}
                      </h4>
                      <p className={`text-[9px] font-mono tracking-[0.2em] mt-1 uppercase ${isExpired ? "text-red-500/60 font-bold" : "text-foreground/30"}`}>
                        {isExpired ? "EXPIRED" : `*${acc.account_number?.slice(-4) || "0000"}`}
                      </p>
                    </div>
                    <div className="flex flex-col items-end shrink-0 pl-2">
                      <span className={`font-mono font-bold text-sm ${isExpired ? "text-red-500/50" : selectedAccountId === acc.account_id ? "text-primary" : "text-foreground"}`}>
                        {new Intl.NumberFormat("en-GB", { style: "currency", currency: acc.currency || "GBP" }).format(acc.balance || 0)}
                      </span>
                      <span className={`text-[8px] font-black uppercase tracking-[0.2em] mt-0.5 ${isExpired ? "text-red-500/30" : "text-foreground/20"}`}>
                        {acc.currency || "GBP"}
                      </span>
                    </div>
                  </div>
                </Card>
              )
            })
          )}
        </div>

        {/* Right Panel: Detailed Profile */}
        <div className="lg:col-span-8 flex flex-col gap-6 overflow-y-auto scrollbar-hide h-full pb-10">
          <h3 className="text-[10px] font-black text-foreground/30 uppercase tracking-[0.4em] px-2 italic shrink-0">
            Account Profile
          </h3>

          {!selectedAccount ? (
            <div className="flex items-center justify-center h-64 border border-dashed border-white/10 rounded-2xl">
              <span className="text-foreground/30 font-mono text-[10px] uppercase tracking-widest">Select an account</span>
            </div>
          ) : (
            <div className="flex flex-col gap-6 w-full">
              {/* Header & Status Card (Options 1, 2 & 4 Combined) */}
              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full">
                <Card className="liquid-glass rounded-2xl p-8 border-none shadow-inner flex flex-col justify-between">
                  <div>
                    <p className="text-[9px] font-black text-primary uppercase tracking-[0.3em] mb-2 flex items-center gap-2">
                      <Database size={12} /> Live Data
                    </p>
                    <h2 className="text-2xl font-black text-foreground uppercase tracking-tight italic mb-1 truncate">
                      {selectedAccount.bank_name}
                    </h2>
                    <p className="text-[11px] font-mono text-foreground/40 uppercase tracking-widest">
                      {selectedAccount.account_number || "**** 0000"} | {selectedAccount.sort_code || "00-00-00"}
                    </p>
                  </div>
                  <div className="mt-8">
                    <p className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.3em] mb-1">
                      Current Balance
                    </p>
                    <div className="text-3xl font-mono text-foreground">
                      {selectedAccount.balance ? new Intl.NumberFormat("en-GB", { style: "currency", currency: selectedAccount.currency || "GBP" }).format(selectedAccount.balance) : <Skeleton className="h-8 w-32 bg-white/5 rounded-lg" />}
                    </div>
                  </div>
                </Card>

                {/* System Status / Authentication Lock */}
                {selectedAccount.consent_status === "200-401" || selectedAccount.consent_status === "200-403" ? (
                  <Card className="liquid-glass rounded-2xl p-8 border border-red-500/30 shadow-inner bg-red-500/5 flex flex-col justify-center items-center text-center gap-4">
                    <div className="w-12 h-12 rounded-full bg-red-500/20 flex items-center justify-center text-red-500 text-xl animate-pulse">
                      ⚠️
                    </div>
                    <div>
                      <h3 className="text-red-500 font-black uppercase italic tracking-tighter text-lg">{isHardRevoked ? "Access Revoked" : "Connection Expired"}</h3>
                      <p className="text-red-500/70 text-[9px] font-mono tracking-widest mt-1">{isHardRevoked ? "BANK HAS REVOKED ACCESS" : "SCA 90-DAY LIMIT REACHED"}</p>
                    </div>
                    <Button
                      onPress={() => handleReauth(selectedAccount.bank_uuid)}
                      isPending={isReauthenticating}
                      className="mt-2 font-black text-[10px] uppercase tracking-widest px-8 h-10 rounded-lg cursor-pointer bg-red-500 text-white hover:bg-red-600 shadow-lg shadow-red-500/20"
                    >
                      Re-authenticate
                    </Button>
                  </Card>
                ) : (
                  <Card className="liquid-glass rounded-2xl p-8 border border-white/5 shadow-inner bg-black/20 flex flex-col gap-6">
                    <div className="flex items-center justify-between">
                      <p className="text-[9px] font-black text-foreground/50 uppercase tracking-[0.3em] flex items-center gap-2">
                        <Activity size={12} /> Sync Status
                      </p>
                      <Button
                        onPress={() => handleReauth(selectedAccount.bank_uuid)}
                        isPending={isReauthenticating}
                        className="flex items-center justify-center gap-2 font-black text-[10px] uppercase tracking-widest rounded-xl px-4 h-8 transition-all cursor-pointer bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 shadow-lg"
                      >
                        <RefreshCcw size={12} className={cn(isReauthenticating && "animate-spin")} /> Reconnect
                      </Button>
                    </div>
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="flex justify-between text-[10px] font-mono text-foreground/40">
                          <span>Data Pipeline</span>
                          <span>Pending</span>
                        </div>
                        <Skeleton className="w-full h-1 bg-white/5 rounded-full" />
                      </div>
                      <div className="space-y-2">
                        <div className="flex justify-between text-[10px] font-mono text-foreground/40">
                          <span>API Quota (Tokens)</span>
                          <span>Calculating</span>
                        </div>
                        <Skeleton className="w-full h-1 bg-white/5 rounded-full" />
                      </div>
                    </div>
                  </Card>
                )}
              </div>

              {/* Visualization Row: Pie Chart & Subscriptions */}
              {(selectedAccount.consent_status === "200-401" || selectedAccount.consent_status === "200-403") && (
                <div className="absolute inset-0 z-50 bg-background/80 backdrop-blur-md flex flex-col items-center justify-center rounded-2xl border border-red-500/20 mt-48">
                  <div className="bg-red-500/10 p-6 rounded-full mb-6 border border-red-500/30 shadow-[0_0_50px_rgba(239,68,68,0.2)]">
                    <Database size={48} className="text-red-500 opacity-80" />
                  </div>
                  <h2 className="text-3xl font-black text-red-500 uppercase tracking-tighter italic mb-2">Access Locked</h2>
                  <p className="text-red-500/60 text-xs font-mono tracking-widest max-w-sm text-center">
                    YOUR BANK REQUIRES RE-AUTHENTICATION BEFORE BUD-AI CAN FETCH NEW TRANSACTIONS OR PROJECT FINANCES.
                  </p>
                  <Button
                    onPress={() => handleReauth(selectedAccount.bank_uuid)}
                    isPending={isReauthenticating}
                    className="mt-8 font-black text-[12px] uppercase tracking-widest px-10 h-14 rounded-xl cursor-pointer bg-red-500 text-white hover:bg-red-600 shadow-[0_0_30px_rgba(239,68,68,0.4)] transition-all hover:scale-105"
                  >
                    Restore Access Now
                  </Button>
                </div>
              )}

              <div className="grid grid-cols-1 md:grid-cols-2 gap-6 w-full h-[400px]">
                <Card className="liquid-glass rounded-xl h-full flex flex-col p-8 overflow-hidden shadow-inner border-[0.5px] border-white/5">
                  <div className="flex justify-between items-start mb-6 shrink-0 z-10">
                    <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                      Expense Distribution
                    </h3>
                  </div>
                  <div className="flex-1 w-full relative min-h-0 flex items-center justify-center -mt-6">
                    {isExpenseFetching ? (
                      <Skeleton className="w-48 h-48 rounded-full bg-white/5" />
                    ) : pieChartConfig ? (
                      <CoreChartEngine config={pieChartConfig} />
                    ) : (
                      <span className="text-[10px] font-mono uppercase tracking-widest text-foreground/40">No Data Available</span>
                    )}
                  </div>
                </Card>

                <Card className="liquid-glass rounded-xl h-full flex flex-col p-8 overflow-hidden shadow-inner border-[0.5px] border-white/5">
                  <div className="flex justify-between items-start mb-6 shrink-0">
                    <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                      Detected Subscriptions
                    </h3>
                  </div>
                  <div className="flex-1 overflow-auto scrollbar-hide">
                    {isSubFetching ? (
                      <div className="space-y-4">
                        {[1, 2, 3].map((i) => (
                          <Skeleton key={i} className="w-full h-16 bg-white/5 rounded-xl" />
                        ))}
                      </div>
                    ) : accountSubscriptions.length > 0 ? (
                      <div className="flex flex-col gap-3">
                        {accountSubscriptions.map((sub: any, idx: number) => (
                          <div key={idx} className="flex justify-between items-center p-4 rounded-xl bg-white/5 border border-white/10 hover:bg-white/10 transition-colors">
                            <div className="flex items-center gap-4">
                              <div className="w-10 h-10 rounded-full bg-primary/20 flex items-center justify-center text-primary font-black uppercase text-xs">
                                {sub.merchant_name.charAt(0)}
                              </div>
                              <div className="flex flex-col">
                                <span className="font-bold text-foreground text-sm uppercase tracking-tight truncate max-w-[120px]">
                                  {sub.merchant_name}
                                </span>
                                <span className="text-[9px] font-mono text-foreground/50 uppercase tracking-widest">
                                  {sub.status === '303-410' ? 'Expired' : 'Active'} • {sub.predicted_frequency}
                                </span>
                              </div>
                            </div>
                            <div className="flex flex-col items-end">
                              <span className="font-mono font-bold text-foreground">
                                {new Intl.NumberFormat("en-GB", { style: "currency", currency: "GBP" }).format(sub.expected_amount)}
                              </span>
                              {sub.status !== '303-410' && (
                                <span className="text-[9px] font-mono text-primary/70 tracking-widest uppercase">
                                  {sub.next_expected_date ? new Date(sub.next_expected_date).toLocaleDateString("en-GB", { day: 'numeric', month: 'short' }) : 'Pending'}
                                </span>
                              )}
                            </div>
                          </div>
                        ))}
                      </div>
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <span className="text-[10px] font-mono uppercase tracking-widest text-foreground/40">No Subscriptions Found</span>
                      </div>
                    )}
                  </div>
                </Card>
              </div>

              {/* Transactions Table */}
              <div className="w-full min-h-[400px] flex-1">
                <Card className="liquid-glass rounded-xl h-full flex flex-col p-8 overflow-hidden shadow-inner border-[0.5px] border-white/5">
                  <div className="flex justify-between items-start mb-6 shrink-0">
                    <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                      Recent Transactions
                    </h3>
                  </div>
                  <div className="flex-1 overflow-auto scrollbar-hide">
                    {isTxFetching ? (
                      <div className="space-y-4">
                        {[1, 2, 3, 4, 5, 6].map((i) => (
                          <Skeleton key={i} className="w-full h-12 bg-white/5 rounded-lg" />
                        ))}
                      </div>
                    ) : recentTransactions.length > 0 ? (
                      <table className="w-full text-left border-collapse">
                        <thead>
                          <tr>
                            <th className="bg-transparent text-[9px] font-black tracking-widest uppercase text-foreground/40 border-b border-white/10 pb-4 font-sans">Date</th>
                            <th className="bg-transparent text-[9px] font-black tracking-widest uppercase text-foreground/40 border-b border-white/10 pb-4 font-sans">Merchant</th>
                            <th className="bg-transparent text-[9px] font-black tracking-widest uppercase text-foreground/40 border-b border-white/10 pb-4 font-sans">Category</th>
                            <th className="bg-transparent text-[9px] font-black tracking-widest uppercase text-foreground/40 border-b border-white/10 pb-4 text-right font-sans">Amount</th>
                          </tr>
                        </thead>
                        <tbody>
                          {recentTransactions.map((tx: any) => (
                            <tr key={tx.transaction_id} className="border-b border-white/5 hover:bg-white/5 transition-colors">
                              <td className="text-foreground/60 text-xs font-mono py-4">{new Date(tx.date).toLocaleDateString("en-GB")}</td>
                              <td className="text-foreground font-bold py-4 truncate max-w-[150px]">{tx.merchant_name || tx.description}</td>
                              <td className="py-4">
                                <div className="flex flex-col gap-1.5 items-start">
                                  <span className={cn("px-2 py-1 rounded-lg text-[9px] font-black uppercase tracking-widest border-[0.5px] shadow-sm", getCategoryTheme(tx.category || "Other"))}>
                                    {tx.category || "Other"}
                                  </span>
                                  {tx.sub_category && (
                                    <span className="text-[9px] text-foreground/40 uppercase tracking-widest font-mono">
                                      {tx.sub_category}
                                    </span>
                                  )}
                                  {tx.tags && tx.tags.length > 0 && (
                                    <div className="flex flex-wrap gap-1 mt-1">
                                      {tx.tags.map((tag: string, idx: number) => {
                                        let tagClass = "bg-white/5 text-foreground/50 border-white/10";
                                        if (tag === "#recurring") tagClass = "bg-purple-500/20 text-purple-400 border-purple-500/40";
                                        if (tag === "#essential" || tag === "#housing") tagClass = "bg-green-500/20 text-green-400 border-green-500/40";
                                        if (tag === "#price-hike") tagClass = "bg-red-500/20 text-red-500 border-red-500/40 animate-pulse";
                                        
                                        return (
                                          <span key={idx} className={cn("px-1.5 py-0.5 rounded text-[8px] font-black uppercase tracking-widest border-[0.5px]", tagClass)}>
                                            {tag}
                                          </span>
                                        );
                                      })}
                                    </div>
                                  )}
                                </div>
                              </td>
                              <td className="font-mono font-bold text-right py-4">
                                <span className={tx.amount > 0 ? "text-green-500" : "text-foreground"}>
                                  {tx.amount > 0 ? "+" : ""}{new Intl.NumberFormat("en-GB", { style: "currency", currency: tx.currency || "GBP" }).format(Math.abs(tx.amount))}
                                </span>
                              </td>
                            </tr>
                          ))}
                        </tbody>
                      </table>
                    ) : (
                      <div className="flex items-center justify-center h-full">
                        <span className="text-[10px] font-mono uppercase tracking-widest text-foreground/40">No Transactions Found</span>
                      </div>
                    )}
                  </div>
                </Card>
              </div>

            </div>
          )}
        </div>
      </div>
    </div>
  );
}
