"use client";

import React, { useState, useMemo } from "react";
import { Button, Skeleton } from "@heroui/react";
import {
  Search,
  ArrowRight,
  RefreshCw,
  CreditCard,
  CheckCircle2,
  Clock,
  Receipt,
} from "lucide-react";
import { useRouter } from "next/navigation";
import { useBudAI } from "@/app/context/AppContext";
import { useTransactions } from "@/lib/hooks";
import { formatCurrency, getCurrencySymbol } from "@/lib/currency";
import { apiFetch } from "@/lib/api";
import { toast } from "@heroui/react";

interface RecentTransactionsCardProps {
  currency?: string;
}

export const RecentTransactionsCard = ({
  currency = "GBP",
}: RecentTransactionsCardProps) => {
  const router = useRouter();
  const { accounts, activeAccountId } = useBudAI();
  const [searchQuery, setSearchQuery] = useState("");
  const [isSyncing, setIsSyncing] = useState(false);

  const activeAccount = useMemo(() => {
    if (!activeAccountId || activeAccountId === "ALL") return accounts[0] || null;
    return accounts.find((a) => a.account_id === activeAccountId) || accounts[0] || null;
  }, [accounts, activeAccountId]);

  const targetAccountId = activeAccount?.account_id || "";
  const activeCurrency = activeAccount?.currency || currency;

  const { data: transactions = [], isLoading, refetch } = useTransactions(
    targetAccountId,
    undefined,
    undefined,
    undefined,
  );

  const filteredTransactions = useMemo(() => {
    if (!transactions) return [];
    if (!searchQuery.trim()) return transactions.slice(0, 8);

    const q = searchQuery.toLowerCase();
    return transactions
      .filter((t) => {
        const desc = (t.description || t.merchant_name || "").toLowerCase();
        const cat = (t.category || "").toLowerCase();
        const amt = String(t.amount || "");
        return desc.includes(q) || cat.includes(q) || amt.includes(q);
      })
      .slice(0, 8);
  }, [transactions, searchQuery]);

  const handleSyncAccount = async () => {
    setIsSyncing(true);
    try {
      const res = await apiFetch("/api/accounts/sync", { method: "POST" }, true);
      if (res.ok) {
        toast("Bank sync initiated. Transactions updated.");
        refetch();
      } else {
        toast("Bank sync request submitted. Waiting for bank settlement.");
      }
    } catch {
      toast("Unable to reach banking provider at this time.", {
        variant: "danger",
      } as any);
    } finally {
      setIsSyncing(false);
    }
  };

  const handleRowClick = (txId: string) => {
    router.push(`/transactions?highlight=${encodeURIComponent(txId)}`);
  };

  return (
    <div className="flex flex-col gap-3.5 p-5 rounded-2xl bg-[#0d0e14]/40 backdrop-blur-xl border border-white/10 shadow-xl">
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/10 dark:border-white/5 pb-3.5">
        <div>
          <h3 className="text-base font-bold text-foreground">
            Recent Transactions
          </h3>
          <p className="text-xs text-muted-foreground mt-0.5">
            Latest payments and verified categories
          </p>
        </div>

        <div className="flex items-center gap-2">
          <div className="relative flex items-center">
            <Search
              size={14}
              className="absolute left-3 text-muted-foreground pointer-events-none"
            />
            <input
              type="text"
              placeholder="Search payments..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="bg-white/40 dark:bg-neutral-800/40 hover:bg-white/60 dark:hover:bg-neutral-800/60 border border-white/10 rounded-xl pl-8 pr-3 py-1.5 text-xs text-foreground placeholder:text-muted-foreground outline-none focus:border-primary/50 transition-colors w-36 sm:w-48"
            />
          </div>

          <Button
            size="sm"
            variant="ghost"
            onPress={() => router.push("/transactions")}
            className="text-xs font-semibold h-8 px-3 rounded-xl border border-white/10 hover:bg-white/10 flex items-center gap-1.5 shrink-0 transition-all cursor-pointer"
          >
            <span>View All</span>
            <ArrowRight size={13} />
          </Button>
        </div>
      </div>

      <div className="flex flex-col gap-2 min-h-[220px]">
        {isLoading ? (
          <div className="flex flex-col gap-2 py-2">
            {[1, 2, 3, 4].map((i) => (
              <div
                key={i}
                className="flex items-center justify-between p-3 rounded-xl border border-white/10 bg-white/30 dark:bg-neutral-800/30"
              >
                <div className="flex items-center gap-3">
                  <Skeleton className="w-9 h-9 rounded-xl" />
                  <div className="flex flex-col gap-1">
                    <Skeleton className="w-36 h-3.5 rounded-md" />
                    <Skeleton className="w-24 h-2.5 rounded-md" />
                  </div>
                </div>
                <Skeleton className="w-20 h-3.5 rounded-md" />
              </div>
            ))}
          </div>
        ) : filteredTransactions.length === 0 ? (
          <div className="flex-1 flex flex-col items-center justify-center p-8 text-center rounded-xl border border-dashed border-white/15 bg-white/30 dark:bg-neutral-800/30 gap-2.5 my-auto">
            <Receipt size={28} className="text-muted-foreground/60" />
            <p className="text-sm font-semibold text-foreground">
              {searchQuery
                ? "No transactions match your search"
                : "No transactions recorded yet"}
            </p>
            <p className="text-xs text-muted-foreground max-w-sm">
              {searchQuery
                ? "Try searching for a different merchant or category name."
                : "Connect your bank account or trigger a sync to pull verified transactions."}
            </p>
            {!searchQuery && (
              <Button
                size="sm"
                onPress={handleSyncAccount}
                isPending={isSyncing}
                className="mt-2 text-xs font-bold bg-primary text-primary-foreground rounded-xl px-4 py-2 flex items-center gap-1.5 shadow-sm"
              >
                <RefreshCw size={13} className={isSyncing ? "animate-spin" : ""} />
                <span>Sync Bank Account</span>
              </Button>
            )}
          </div>
        ) : (
          filteredTransactions.map((tx) => {
            const txId =
              tx.transaction_id ||
              tx.transaction_uuid ||
              (tx as any).id ||
              Math.random().toString();
            const desc =
              tx.description || tx.merchant_name || "Payment Transaction";
            const cat = tx.category || tx.Category || "General";
            const amt = tx.amount ?? (tx as any).Amount ?? 0;
            const isCredit = amt > 0;
            const dateStr = tx.date || tx.timestamp || "";
            const formattedDate = dateStr
              ? new Date(dateStr).toLocaleDateString("en-GB", {
                  month: "short",
                  day: "numeric",
                })
              : "";
            const isPending =
              tx.status?.toLowerCase() === "pending" ||
              cat.toLowerCase() === "pending";

            return (
              <div
                key={txId}
                onClick={() => handleRowClick(txId)}
                className="flex items-center justify-between p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 hover:bg-white/60 dark:hover:bg-neutral-800/60 border border-white/10 dark:border-white/5 hover:border-white/20 transition-all cursor-pointer group shadow-xs"
              >
                <div className="flex items-center gap-3 min-w-0">
                  <div className="w-9 h-9 rounded-xl bg-white/60 dark:bg-neutral-700/50 border border-white/10 flex items-center justify-center shrink-0 group-hover:scale-105 transition-transform shadow-xs">
                    <CreditCard size={15} className="text-muted-foreground" />
                  </div>
                  <div className="min-w-0">
                    <span className="text-xs md:text-sm font-bold text-foreground truncate block">
                      {desc}
                    </span>
                    <div className="flex items-center gap-2 text-xs text-muted-foreground mt-0.5">
                      <span className="truncate max-w-[130px] font-medium">{cat}</span>
                      {formattedDate && <span className="opacity-70">· {formattedDate}</span>}
                    </div>
                  </div>
                </div>

                <div className="flex items-center gap-2.5 shrink-0">
                  {isPending ? (
                    <span className="flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-md bg-amber-500/15 text-amber-500 border border-amber-500/25">
                      <Clock size={11} />
                      <span>Pending</span>
                    </span>
                  ) : (
                    <span className="flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-md bg-foreground/10 text-muted-foreground">
                      <CheckCircle2 size={11} />
                      <span>Settled</span>
                    </span>
                  )}
                  <span
                    className={`text-xs md:text-sm font-bold font-mono ${
                      isCredit
                        ? "text-emerald-500"
                        : "text-foreground"
                    }`}
                  >
                    {isCredit ? "+" : ""}
                    {formatCurrency(amt, activeCurrency)}
                  </span>
                </div>
              </div>
            );
          })
        )}
      </div>
    </div>
  );
};
