"use client";

import React, { useState, useMemo, useEffect, useRef } from "react";
import { useSearchParams, useRouter } from "next/navigation";
import {
  Search,
  Download,
  Filter,
  RefreshCw,
  CreditCard,
  CheckCircle2,
  Clock,
  ArrowUpDown,
  Tag,
  ChevronDown,
  Building2,
  Calendar,
} from "lucide-react";
import { Button, Dropdown, Skeleton, toast, Modal } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { useTransactions } from "@/lib/hooks";
import { formatCurrency, getCurrencySymbol } from "@/lib/currency";
import { apiFetch } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { Transaction } from "@/types";

const STANDARD_CATEGORIES = [
  "All",
  "Income",
  "Housing",
  "Food & Dining",
  "Transportation",
  "Utilities",
  "Entertainment & Lifestyle",
  "Subscriptions & Digital Services",
  "Shopping & Retail",
  "Healthcare",
  "Transfers & Payments",
  "Fees & Charges",
  "Savings & Investments",
  "Uncategorized",
];

export default function TransactionsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const highlightId = searchParams.get("highlight");
  const queryClient = useQueryClient();
  const { accounts, activeAccountId, setActiveAccountId } = useBudAI();

  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [timeRange, setTimeRange] = useState<"30d" | "90d" | "180d" | "all">("90d");
  const [isSyncing, setIsSyncing] = useState(false);
  const [editingTx, setEditingTx] = useState<Transaction | null>(null);
  const [isUpdatingCategory, setIsUpdatingCategory] = useState(false);

  const highlightedRowRef = useRef<HTMLTableRowElement | null>(null);

  const activeAccount = useMemo(() => {
    if (!activeAccountId || activeAccountId === "ALL") return accounts[0] || null;
    return accounts.find((a) => a.account_id === activeAccountId) || accounts[0] || null;
  }, [accounts, activeAccountId]);

  const activeCurrency = activeAccount?.currency || "GBP";
  const targetAccountId = activeAccount?.account_id || "";

  const { fromDate, toDate } = useMemo(() => {
    const end = new Date();
    const start = new Date();
    if (timeRange === "30d") start.setDate(end.getDate() - 30);
    else if (timeRange === "90d") start.setDate(end.getDate() - 90);
    else if (timeRange === "180d") start.setDate(end.getDate() - 180);
    else start.setFullYear(end.getFullYear() - 5);

    return {
      fromDate: start.toISOString().split("T")[0],
      toDate: end.toISOString().split("T")[0],
    };
  }, [timeRange]);

  const { data: transactions = [], isLoading, refetch } = useTransactions(
    targetAccountId,
    timeRange === "all" ? undefined : fromDate,
    timeRange === "all" ? undefined : toDate,
  );

  const filteredTransactions = useMemo(() => {
    let list = Array.isArray(transactions) ? transactions : [];

    if (selectedCategory !== "All") {
      list = list.filter((t) => {
        const cat = (t.category || t.Category || "Uncategorized").toLowerCase();
        return cat === selectedCategory.toLowerCase();
      });
    }

    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      list = list.filter((t) => {
        const desc = (t.description || t.merchant_name || "").toLowerCase();
        const cat = (t.category || t.Category || "").toLowerCase();
        const amt = String(t.amount || "");
        return desc.includes(q) || cat.includes(q) || amt.includes(q);
      });
    }

    return list;
  }, [transactions, selectedCategory, searchQuery]);

  useEffect(() => {
    if (highlightId && highlightedRowRef.current) {
      highlightedRowRef.current.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [highlightId, filteredTransactions]);

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

  const handleExportCSV = () => {
    if (!filteredTransactions.length) {
      toast("No transactions to export.", { variant: "danger" } as any);
      return;
    }

    const headers = ["Date", "Description", "Category", "Amount", "Currency", "Status"];
    const rows = filteredTransactions.map((t) => [
      t.date || t.timestamp || "",
      `"${(t.description || t.merchant_name || "").replace(/"/g, '""')}"`,
      `"${(t.category || t.Category || "Uncategorized").replace(/"/g, '""')}"`,
      t.amount ?? 0,
      t.currency || activeCurrency,
      t.status || "settled",
    ]);

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute(
      "download",
      `BudAI_Transactions_${new Date().toISOString().split("T")[0]}.csv`,
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast("Transactions exported to CSV.");
  };

  const handleUpdateCategory = async (tx: Transaction, newCategory: string) => {
    const txId = tx.transaction_uuid || tx.transaction_id || (tx as any).id;
    if (!txId) return;

    setIsUpdatingCategory(true);
    try {
      const res = await apiFetch(
        "/api/categorizer/labels",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            transaction_uuid: String(txId),
            corrected_label: newCategory,
            retrain_model: true,
          }),
        },
        true,
      );

      if (res.ok) {
        toast(`Transaction category updated to ${newCategory}.`);
        queryClient.invalidateQueries({ queryKey: ["transactions"] });
        setEditingTx(null);
      } else {
        toast("Failed to update category.", { variant: "danger" } as any);
      }
    } catch {
      toast("Network error updating transaction category.", {
        variant: "danger",
      } as any);
    } finally {
      setIsUpdatingCategory(false);
    }
  };

  return (
    <div className="w-full h-full overflow-y-auto scrollbar-hide px-4 sm:px-6 lg:px-8 py-6 pb-28">
      <div className="max-w-7xl mx-auto flex flex-col gap-6">
        <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-4 border-b border-border/50 pb-4">
          <div>
            <h1 className="text-xl font-bold text-foreground">Transactions</h1>
            <p className="text-xs text-muted-foreground mt-0.5">
              Verified payment history, categories, and audit records
            </p>
          </div>

          <div className="flex items-center gap-2">
            <Button
              size="sm"
              variant="ghost"
              onPress={handleExportCSV}
              className="text-xs font-semibold h-8 px-3 rounded-xl border border-border hover:bg-foreground/5 flex items-center gap-1.5"
            >
              <Download size={13} />
              <span>Export CSV</span>
            </Button>
            <Button
              size="sm"
              onPress={handleSyncAccount}
              isPending={isSyncing}
              className="text-xs font-bold h-8 px-3 rounded-xl bg-primary text-primary-foreground hover:opacity-90 flex items-center gap-1.5"
            >
              <RefreshCw size={13} className={isSyncing ? "animate-spin" : ""} />
              <span>Sync</span>
            </Button>
          </div>
        </div>

        <div className="flex flex-col md:flex-row items-stretch md:items-center justify-between gap-3">
          <div className="relative flex-1 max-w-md flex items-center">
            <Search
              size={14}
              className="absolute left-3 text-muted-foreground pointer-events-none"
            />
            <input
              type="text"
              placeholder="Search by merchant, category, or amount..."
              value={searchQuery}
              onChange={(e) => setSearchQuery(e.target.value)}
              className="w-full bg-foreground/5 hover:bg-foreground/10 border border-border rounded-xl pl-8 pr-3 py-2 text-xs text-foreground placeholder:text-muted-foreground outline-none focus:border-primary/50 transition-colors"
            />
          </div>

          <div className="flex items-center gap-2 overflow-x-auto py-1 scrollbar-hide">
            <div className="flex items-center bg-foreground/5 border border-border rounded-xl p-0.5">
              {(["30d", "90d", "180d", "all"] as const).map((r) => (
                <button
                  key={r}
                  onClick={() => setTimeRange(r)}
                  className={`px-2.5 py-1 rounded-lg text-[11px] font-bold transition-all ${
                    timeRange === r
                      ? "bg-primary text-primary-foreground shadow-xs"
                      : "text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {r === "all" ? "All Time" : `${r.replace("d", "")} Days`}
                </button>
              ))}
            </div>
          </div>
        </div>

        <div className="flex items-center gap-1.5 overflow-x-auto pb-1 scrollbar-hide">
          {STANDARD_CATEGORIES.map((cat) => (
            <button
              key={cat}
              onClick={() => setSelectedCategory(cat)}
              className={`px-3 py-1 rounded-full text-xs font-semibold whitespace-nowrap transition-all border ${
                selectedCategory === cat
                  ? "bg-primary/10 border-primary/30 text-primary"
                  : "bg-foreground/5 border-border text-muted-foreground hover:text-foreground hover:bg-foreground/10"
              }`}
            >
              {cat}
            </button>
          ))}
        </div>

        <div className="bg-[#0d0e14]/40 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-foreground/5 text-muted-foreground font-semibold border-b border-border text-[11px] uppercase tracking-wider">
                <tr>
                  <th className="py-3 px-4">Date</th>
                  <th className="py-3 px-4">Description / Merchant</th>
                  <th className="py-3 px-4">Category</th>
                  <th className="py-3 px-4">Status</th>
                  <th className="py-3 px-4 text-right">Amount</th>
                  <th className="py-3 px-4 text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40">
                {isLoading ? (
                  [1, 2, 3, 4, 5, 6].map((i) => (
                    <tr key={i} className="animate-pulse">
                      <td className="py-3 px-4">
                        <Skeleton className="h-3 w-16 rounded" />
                      </td>
                      <td className="py-3 px-4">
                        <Skeleton className="h-3 w-40 rounded" />
                      </td>
                      <td className="py-3 px-4">
                        <Skeleton className="h-3 w-24 rounded" />
                      </td>
                      <td className="py-3 px-4">
                        <Skeleton className="h-3 w-16 rounded" />
                      </td>
                      <td className="py-3 px-4 text-right">
                        <Skeleton className="h-3 w-16 rounded ml-auto" />
                      </td>
                      <td className="py-3 px-4 text-center">
                        <Skeleton className="h-3 w-10 rounded mx-auto" />
                      </td>
                    </tr>
                  ))
                ) : filteredTransactions.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-12 text-center">
                      <div className="flex flex-col items-center justify-center gap-2">
                        <CreditCard size={28} className="text-muted-foreground/50" />
                        <p className="text-sm font-semibold text-foreground">
                          No transactions found
                        </p>
                        <p className="text-xs text-muted-foreground max-w-sm">
                          {searchQuery || selectedCategory !== "All"
                            ? "Try adjusting your filters or search terms."
                            : "Connect your bank account or trigger a manual sync to pull recent activity."}
                        </p>
                      </div>
                    </td>
                  </tr>
                ) : (
                  filteredTransactions.map((tx) => {
                    const txId =
                      tx.transaction_id ||
                      tx.transaction_uuid ||
                      (tx as any).id ||
                      Math.random().toString();
                    const isHighlighted = highlightId === txId;
                    const desc =
                      tx.description || tx.merchant_name || "Payment";
                    const cat = tx.category || tx.Category || "Uncategorized";
                    const amt = tx.amount ?? (tx as any).Amount ?? 0;
                    const isCredit = amt > 0;
                    const dateStr = tx.date || tx.timestamp || "";
                    const formattedDate = dateStr
                      ? new Date(dateStr).toLocaleDateString("en-GB", {
                          year: "numeric",
                          month: "short",
                          day: "numeric",
                        })
                      : "—";
                    const isPending =
                      tx.status?.toLowerCase() === "pending" ||
                      cat.toLowerCase() === "pending";

                    return (
                      <tr
                        key={txId}
                        ref={isHighlighted ? highlightedRowRef : null}
                        className={`transition-all hover:bg-foreground/5 ${
                          isHighlighted
                            ? "bg-primary/10 ring-1 ring-primary/40 font-semibold"
                            : ""
                        }`}
                      >
                        <td className="py-3 px-4 font-mono text-muted-foreground whitespace-nowrap">
                          {formattedDate}
                        </td>
                        <td className="py-3 px-4 font-semibold text-foreground max-w-[240px] truncate">
                          {desc}
                        </td>
                        <td className="py-3 px-4 whitespace-nowrap">
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-foreground/5 border border-border">
                            <Tag size={10} className="text-muted-foreground" />
                            <span>{cat}</span>
                          </span>
                        </td>
                        <td className="py-3 px-4 whitespace-nowrap">
                          {isPending ? (
                            <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-md bg-amber-500/10 text-amber-500 border border-amber-500/20">
                              <Clock size={10} />
                              <span>Pending</span>
                            </span>
                          ) : (
                            <span className="inline-flex items-center gap-1 text-[10px] font-semibold px-2 py-0.5 rounded-md bg-foreground/10 text-muted-foreground">
                              <CheckCircle2 size={10} />
                              <span>Settled</span>
                            </span>
                          )}
                        </td>
                        <td className="py-3 px-4 text-right font-mono font-bold whitespace-nowrap">
                          <span
                            className={
                              isCredit
                                ? "text-emerald-500"
                                : "text-foreground"
                            }
                          >
                            {isCredit ? "+" : ""}
                            {formatCurrency(amt, tx.currency || activeCurrency)}
                          </span>
                        </td>
                        <td className="py-3 px-4 text-center whitespace-nowrap">
                          <Dropdown>
                            <Button
                              variant="ghost"
                              className="text-[11px] font-semibold text-primary hover:underline cursor-pointer p-0 h-auto min-w-0 bg-transparent shadow-none"
                            >
                              Re-classify
                            </Button>
                            <Dropdown.Popover className="bg-popover border border-border shadow-2xl rounded-xl w-56 p-1.5 z-50">
                              <Dropdown.Menu
                                onAction={(key) =>
                                  handleUpdateCategory(tx, String(key))
                                }
                              >
                                {STANDARD_CATEGORIES.filter(
                                  (c) => c !== "All",
                                ).map((c) => (
                                  <Dropdown.Item
                                    key={c}
                                    id={c}
                                    textValue={c}
                                    className="px-3 py-1.5 rounded-lg text-xs hover:bg-foreground/5 cursor-pointer"
                                  >
                                    {c}
                                  </Dropdown.Item>
                                ))}
                              </Dropdown.Menu>
                            </Dropdown.Popover>
                          </Dropdown>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </div>
      </div>
    </div>
  );
}
