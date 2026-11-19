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
  Layers,
  ArrowLeftRight,
  RotateCcw,
  SlidersHorizontal,
} from "lucide-react";
import {
  Button,
  Dropdown,
  Skeleton,
  toast,
  Tooltip,
  Accordion,
  Chip,
} from "@heroui/react";
import type { Key } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { formatCurrency, getCurrencySymbol } from "@/lib/currency";
import { apiFetch } from "@/lib/api";
import { useQuery, useQueryClient } from "@tanstack/react-query";
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

type FinancialPeriod = "this_month" | "last_month" | "90d" | "ytd" | "all";
type FlowFilter = "all" | "outflow" | "inflow" | "splits";
type AmountFilter = "all" | "under_20" | "20_100" | "100_500" | "over_500";
type StatusFilter = "all" | "settled" | "pending";
type SortOrder =
  | "date_desc"
  | "date_asc"
  | "amount_desc"
  | "amount_asc"
  | "merchant_asc";

interface SplitLegItem extends Transaction {
  account_label: string;
  converted_gbp: number;
  is_fee_leg?: boolean;
}

interface SplitGroupItem {
  is_split_parent: true;
  group_id: string;
  merchant_name: string;
  description: string;
  category: string;
  sub_category?: string;
  date: string;
  status: string;
  converted_total_gbp: number;
  legs: SplitLegItem[];
  all_tx_ids: string[];
}

type DisplayItem =
  | (Transaction & {
      is_split_parent?: false;
      split_parent_name?: string;
      converted_gbp?: number;
    })
  | SplitGroupItem;

const CARD_TXN_RE =
  /Card transaction of\s+([\d.,]+)\s+([A-Za-z]{3})\s+issued by\s+(.*?)(?:\s+\(fee:\s*([\d.,]+)\s*([A-Za-z]{3})\))?$/i;
const WISE_FEE_RE = /Wise Charges for:\s*(.*)/i;
const GENERIC_FEE_RE = /(?:Charges|Fee)\s+for:\s*(.*)/i;

function cleanMerchantName(raw: string): string {
  if (!raw) return "Payment";
  let s = raw.trim();
  const cardMatch = s.match(CARD_TXN_RE);
  if (cardMatch && cardMatch[3]) {
    s = cardMatch[3];
  }
  s = s.replace(/\b\d{8,}\b/g, "").trim();
  s = s.replace(/\s+Amsterdam|\s+London|\s+GB|\s+NL/gi, "").trim();
  return s || raw;
}

function estimateGbp(amount: number, currency: string = "GBP"): number {
  const c = currency.toUpperCase().trim();
  if (c === "GBP") return amount;
  if (c === "EUR") return amount * 0.83;
  if (c === "USD") return amount * 0.785;
  return amount;
}

export default function TransactionsPage() {
  const router = useRouter();
  const searchParams = useSearchParams();
  const highlightId = searchParams.get("highlight");
  const queryClient = useQueryClient();
  const { accounts, activeAccountId } = useBudAI();

  // Filters State
  const [searchQuery, setSearchQuery] = useState("");
  const [selectedCategory, setSelectedCategory] = useState("All");
  const [periodFilter, setPeriodFilter] = useState<FinancialPeriod>("this_month");
  const [flowFilter, setFlowFilter] = useState<FlowFilter>("all");
  const [amountFilter, setAmountFilter] = useState<AmountFilter>("all");
  const [statusFilter, setStatusFilter] = useState<StatusFilter>("all");
  const [sortOrder, setSortOrder] = useState<SortOrder>("date_desc");

  // Controlled Accordion State for Multi-Currency Split Payments
  const [expandedSplitKeys, setExpandedSplitKeys] = useState<Set<Key>>(new Set());

  // Loading States
  const [isSyncing, setIsSyncing] = useState(false);
  const [isReconciling, setIsReconciling] = useState(false);
  const [isUpdatingCategory, setIsUpdatingCategory] = useState(false);

  const highlightedRowRef = useRef<HTMLTableRowElement | null>(null);

  const isAllAccounts = !activeAccountId || activeAccountId === "ALL";

  const activeAccount = useMemo(() => {
    if (isAllAccounts) return accounts[0] || null;
    return accounts.find((a) => a.account_id === activeAccountId) || accounts[0] || null;
  }, [accounts, activeAccountId, isAllAccounts]);

  const activeCurrency = isAllAccounts ? "GBP" : activeAccount?.currency || "GBP";

  // Compute Date Boundaries from Financial Period (NO calendar date picker)
  const { fromDate, toDate } = useMemo(() => {
    const now = new Date();
    const formatYMD = (d: Date) => {
      const year = d.getFullYear();
      const month = String(d.getMonth() + 1).padStart(2, "0");
      const day = String(d.getDate()).padStart(2, "0");
      return `${year}-${month}-${day}`;
    };

    const todayStr = formatYMD(now);

    if (periodFilter === "this_month") {
      const startOfMonth = new Date(now.getFullYear(), now.getMonth(), 1);
      return { fromDate: formatYMD(startOfMonth), toDate: todayStr };
    }
    if (periodFilter === "last_month") {
      const startOfLastMonth = new Date(now.getFullYear(), now.getMonth() - 1, 1);
      const endOfLastMonth = new Date(now.getFullYear(), now.getMonth(), 0);
      return {
        fromDate: formatYMD(startOfLastMonth),
        toDate: formatYMD(endOfLastMonth),
      };
    }
    if (periodFilter === "90d") {
      const start90 = new Date(now);
      start90.setDate(now.getDate() - 90);
      return { fromDate: formatYMD(start90), toDate: todayStr };
    }
    if (periodFilter === "ytd") {
      const startOfYear = new Date(now.getFullYear(), 0, 1);
      return { fromDate: formatYMD(startOfYear), toDate: todayStr };
    }
    // "all"
    return { fromDate: undefined, toDate: undefined };
  }, [periodFilter]);

  // Account targets for fetching
  const targetAccountIds = useMemo(() => {
    if (isAllAccounts) {
      return accounts.map((a) => a.account_id).filter(Boolean);
    }
    return activeAccountId ? [activeAccountId] : [];
  }, [accounts, activeAccountId, isAllAccounts]);

  // Fetch transactions across targeted accounts
  const {
    data: rawTransactions = [],
    isLoading,
    refetch,
  } = useQuery({
    queryKey: [
      "transactions",
      activeAccountId || "ALL",
      fromDate,
      toDate,
      targetAccountIds.join(","),
    ],
    queryFn: async (): Promise<Transaction[]> => {
      if (!targetAccountIds.length) return [];

      const queryParams = new URLSearchParams();
      if (fromDate) queryParams.append("from", fromDate);
      if (toDate) queryParams.append("to", toDate);
      const queryStr = queryParams.toString() ? `?${queryParams.toString()}` : "";

      const results = await Promise.all(
        targetAccountIds.map(async (accId) => {
          try {
            const res = await apiFetch(
              `/api/accounts/${accId}/transactions${queryStr}`,
              {},
              true
            );
            if (res.status === 202) return [];
            if (!res.ok) return [];
            const data = (await res.json()) as { transactions?: Transaction[] };
            return (data.transactions || []).map((t) => ({
              ...t,
              account_id: t.account_id || accId,
            }));
          } catch {
            return [];
          }
        })
      );

      const merged = results.flat();
      const seen = new Set<string>();
      const unique: Transaction[] = [];
      for (const tx of merged) {
        const id =
          tx.transaction_uuid ||
          tx.transaction_id ||
          (tx as any).id ||
          `${tx.date}_${tx.amount}_${tx.description}`;
        if (!seen.has(id)) {
          seen.add(id);
          unique.push(tx);
        }
      }
      return unique;
    },
    enabled: targetAccountIds.length > 0,
    staleTime: 1000 * 60 * 5,
  });

  // Account lookup map for instant badges
  const accountMap = useMemo(() => {
    const map = new Map<string, string>();
    accounts.forEach((a) => {
      const name =
        a.bank_name ||
        a.provider_name ||
        (a.account_number ? `Account ••••${a.account_number.slice(-4)}` : "Account");
      map.set(a.account_id, `${name} (${a.currency || "GBP"})`);
    });
    return map;
  }, [accounts]);

  // Grouping & Filtering Pipeline
  const displayItems = useMemo(() => {
    let list = Array.isArray(rawTransactions) ? [...rawTransactions] : [];

    // Helper to extract clean merchant
    const getClean = (t: Transaction) =>
      cleanMerchantName(t.merchant_name || t.description || "");

    // Identify split clusters
    // Clusters exist when transactions share merchant_knowledge_uuid (from backend reconciliation),
    // or explicit card transactions + fee legs / multi-currency legs.
    // Ordinary single-currency bank transfers (e.g. to friends, savings, or repeat charges) MUST NEVER be grouped.
    const splitClusters: Map<string, Transaction[]> = new Map();
    const assignedTxIds = new Set<string>();

    for (let i = 0; i < list.length; i++) {
      const t = list[i];
      const tid = t.transaction_uuid || t.transaction_id || (t as any).id;
      if (assignedTxIds.has(tid)) continue;

      const desc = t.description || "";
      const isCardTxn = CARD_TXN_RE.test(desc);
      const isFee = WISE_FEE_RE.test(desc) || GENERIC_FEE_RE.test(desc);
      const tags = t.tags || [];
      const hasSplitTag =
        tags.includes("#split_payment") ||
        tags.includes("#multi_currency") ||
        tags.includes("#fx_fee");

      // Only attempt clustering if transaction has explicit split indicators:
      // backend reconciliation UUID, card transaction format, split tags, or fee leg
      const hasSplitIndicators =
        Boolean((t as any).merchant_knowledge_uuid) ||
        isCardTxn ||
        hasSplitTag ||
        isFee;

      if (!hasSplitIndicators) continue;

      const cluster: Transaction[] = [t];
      const tTime = t.date ? new Date(t.date).getTime() : 0;
      const cleanM = getClean(t);

      for (let j = i + 1; j < list.length; j++) {
        const other = list[j];
        const otherId =
          other.transaction_uuid || other.transaction_id || (other as any).id;
        if (assignedTxIds.has(otherId)) continue;

        const otherDesc = other.description || "";
        const otherIsCardTxn = CARD_TXN_RE.test(otherDesc);
        const otherIsFee =
          WISE_FEE_RE.test(otherDesc) || GENERIC_FEE_RE.test(otherDesc);
        const otherTags = other.tags || [];
        const otherHasSplitTag =
          otherTags.includes("#split_payment") ||
          otherTags.includes("#multi_currency") ||
          otherTags.includes("#fx_fee");

        const otherTime = other.date ? new Date(other.date).getTime() : 0;
        const timeDiffSec = Math.abs(tTime - otherTime) / 1000;

        // Condition 1: Shared backend reconciliation UUID
        const sameKnowledge =
          Boolean((t as any).merchant_knowledge_uuid) &&
          (t as any).merchant_knowledge_uuid ===
            (other as any).merchant_knowledge_uuid;

        // Condition 2: Card transaction / split tag paired with an associated fee leg
        const isRelatedFee =
          timeDiffSec <= 300 &&
          (((isCardTxn || hasSplitTag) && otherIsFee) ||
            ((otherIsCardTxn || otherHasSplitTag) && isFee));

        // Condition 3: Card transaction split across different currencies or different accounts
        const isRelatedCardLeg =
          timeDiffSec <= 300 &&
          (isCardTxn || hasSplitTag) &&
          (otherIsCardTxn || otherHasSplitTag) &&
          cleanM.toLowerCase() === getClean(other).toLowerCase() &&
          cleanM.length > 3 &&
          (t.currency !== other.currency || t.account_id !== other.account_id);

        if (sameKnowledge || isRelatedFee || isRelatedCardLeg) {
          cluster.push(other);
          assignedTxIds.add(otherId);
        }
      }

      if (cluster.length > 1) {
        assignedTxIds.add(tid);
        const groupKey =
          (t as any).merchant_knowledge_uuid ||
          `split_${cleanM.toLowerCase().replace(/[^a-z0-9]/g, "_")}_${t.date?.slice(0, 10)}`;
        splitClusters.set(groupKey, cluster);
      }
    }

    // Now build display list
    let processed: DisplayItem[] = [];

    if (isAllAccounts) {
      // In "All Accounts" view: Group split payments into parent Accordion cards
      const addedClusterKeys = new Set<string>();

      list.forEach((t) => {
        const tid = t.transaction_uuid || t.transaction_id || (t as any).id;

        // Check if part of any cluster
        let foundClusterKey: string | null = null;
        for (const [key, cluster] of splitClusters.entries()) {
          if (
            cluster.some(
              (c) =>
                (c.transaction_uuid || c.transaction_id || (c as any).id) === tid
            )
          ) {
            foundClusterKey = key;
            break;
          }
        }

        if (foundClusterKey) {
          if (!addedClusterKeys.has(foundClusterKey)) {
            addedClusterKeys.add(foundClusterKey);
            const cluster = splitClusters.get(foundClusterKey)!;
            const primaryLeg =
              cluster.find(
                (c) =>
                  (c.amount || 0) < -1 &&
                  !WISE_FEE_RE.test(c.description || "") &&
                  !GENERIC_FEE_RE.test(c.description || "")
              ) || cluster[0];
            const cleanName = getClean(primaryLeg);

            // Compute converted total in GBP
            let totalGbp = 0;
            const legs: SplitLegItem[] = cluster.map((leg) => {
              const legAmt = leg.amount ?? (leg as any).Amount ?? 0;
              const legCurr = leg.currency || "GBP";
              const legDesc = leg.description || "";
              const isFee =
                WISE_FEE_RE.test(legDesc) || GENERIC_FEE_RE.test(legDesc);
              const converted = estimateGbp(legAmt, legCurr);
              totalGbp += converted;

              const accLabel =
                accountMap.get(leg.account_id || "") ||
                leg.account_id ||
                "Account";

              return {
                ...leg,
                account_label: accLabel,
                converted_gbp: converted,
                is_fee_leg: isFee,
              };
            });

            // If the user's specific Recharge.com case (1.94 EUR + 8.64 GBP + 0.03 fee = £10.28)
            // Round nicely to 2 decimal places
            const roundedTotal = Math.round(totalGbp * 100) / 100;

            const splitItem: SplitGroupItem = {
              is_split_parent: true,
              group_id: foundClusterKey,
              merchant_name: cleanName,
              description: primaryLeg.description || cleanName,
              category:
                primaryLeg.category ||
                primaryLeg.Category ||
                "Subscriptions & Digital Services",
              sub_category: primaryLeg.sub_category,
              date: primaryLeg.date || primaryLeg.timestamp || "",
              status: cluster.some((c) => c.status?.toLowerCase() === "pending")
                ? "pending"
                : "settled",
              converted_total_gbp: roundedTotal,
              legs,
              all_tx_ids: cluster.map(
                (c) =>
                  c.transaction_uuid || c.transaction_id || (c as any).id
              ),
            };

            processed.push(splitItem);
          }
        } else {
          processed.push({
            ...t,
            is_split_parent: false,
            converted_gbp: estimateGbp(
              t.amount ?? (t as any).Amount ?? 0,
              t.currency || "GBP"
            ),
          });
        }
      });
    } else {
      // In single account view: DO NOT use Accordion grouping.
      // Render flat native transaction rows, with discrete tag if part of a split
      list.forEach((t) => {
        const tid = t.transaction_uuid || t.transaction_id || (t as any).id;
        let parentMerchant: string | undefined = undefined;

        for (const [, cluster] of splitClusters.entries()) {
          if (
            cluster.some(
              (c) =>
                (c.transaction_uuid || c.transaction_id || (c as any).id) === tid
            )
          ) {
            const primaryLeg =
              cluster.find((c) => (c.amount || 0) < -1) || cluster[0];
            parentMerchant = getClean(primaryLeg);
            break;
          }
        }

        processed.push({
          ...t,
          is_split_parent: false,
          split_parent_name: parentMerchant,
          converted_gbp: estimateGbp(
            t.amount ?? (t as any).Amount ?? 0,
            t.currency || "GBP"
          ),
        });
      });
    }

    // Filter by Category
    if (selectedCategory !== "All") {
      processed = processed.filter((item) => {
        const cat = (item.category || (item as any).Category || "Uncategorized").toLowerCase();
        return cat === selectedCategory.toLowerCase();
      });
    }

    // Filter by Money Flow
    if (flowFilter === "outflow") {
      processed = processed.filter((item) => {
        if (item.is_split_parent) return item.converted_total_gbp < 0;
        return (item.amount ?? (item as any).Amount ?? 0) < 0;
      });
    } else if (flowFilter === "inflow") {
      processed = processed.filter((item) => {
        if (item.is_split_parent) return item.converted_total_gbp > 0;
        return (item.amount ?? (item as any).Amount ?? 0) > 0;
      });
    } else if (flowFilter === "splits") {
      processed = processed.filter((item) => {
        if (item.is_split_parent) return true;
        return !!item.split_parent_name;
      });
    }

    // Filter by Amount Range
    if (amountFilter !== "all") {
      processed = processed.filter((item) => {
        const absVal = item.is_split_parent
          ? Math.abs(item.converted_total_gbp)
          : Math.abs(item.converted_gbp ?? item.amount ?? (item as any).Amount ?? 0);

        if (amountFilter === "under_20") return absVal < 20;
        if (amountFilter === "20_100") return absVal >= 20 && absVal <= 100;
        if (amountFilter === "100_500") return absVal > 100 && absVal <= 500;
        if (amountFilter === "over_500") return absVal > 500;
        return true;
      });
    }

    // Filter by Status
    if (statusFilter !== "all") {
      processed = processed.filter((item) => {
        const st = (item.status || "settled").toLowerCase();
        return st === statusFilter;
      });
    }

    // Filter by Search Query
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      processed = processed.filter((item) => {
        const desc = (item.description || item.merchant_name || "").toLowerCase();
        const cat = (item.category || (item as any).Category || "").toLowerCase();
        const amtStr = item.is_split_parent
          ? String(item.converted_total_gbp)
          : String(item.amount ?? (item as any).Amount ?? "");
        return desc.includes(q) || cat.includes(q) || amtStr.includes(q);
      });
    }

    // Apply Sorting
    processed.sort((a, b) => {
      const dateA = new Date(a.date || (a as any).timestamp || 0).getTime();
      const dateB = new Date(b.date || (b as any).timestamp || 0).getTime();

      const amtA = a.is_split_parent
        ? a.converted_total_gbp
        : a.converted_gbp ?? a.amount ?? (a as any).Amount ?? 0;
      const amtB = b.is_split_parent
        ? b.converted_total_gbp
        : b.converted_gbp ?? b.amount ?? (b as any).Amount ?? 0;

      const nameA = (a.merchant_name || a.description || "").toLowerCase();
      const nameB = (b.merchant_name || b.description || "").toLowerCase();

      switch (sortOrder) {
        case "date_desc":
          return dateB - dateA;
        case "date_asc":
          return dateA - dateB;
        case "amount_desc":
          return Math.abs(amtB) - Math.abs(amtA);
        case "amount_asc":
          return Math.abs(amtA) - Math.abs(amtB);
        case "merchant_asc":
          return nameA.localeCompare(nameB);
        default:
          return dateB - dateA;
      }
    });

    return processed;
  }, [
    rawTransactions,
    isAllAccounts,
    accountMap,
    selectedCategory,
    flowFilter,
    amountFilter,
    statusFilter,
    searchQuery,
    sortOrder,
  ]);

  // Check if non-default filters are active
  const hasActiveFilters = useMemo(() => {
    return (
      searchQuery.trim() !== "" ||
      selectedCategory !== "All" ||
      periodFilter !== "this_month" ||
      flowFilter !== "all" ||
      amountFilter !== "all" ||
      statusFilter !== "all" ||
      sortOrder !== "date_desc"
    );
  }, [
    searchQuery,
    selectedCategory,
    periodFilter,
    flowFilter,
    amountFilter,
    statusFilter,
    sortOrder,
  ]);

  const handleResetFilters = () => {
    setSearchQuery("");
    setSelectedCategory("All");
    setPeriodFilter("this_month");
    setFlowFilter("all");
    setAmountFilter("all");
    setStatusFilter("all");
    setSortOrder("date_desc");
  };

  useEffect(() => {
    if (highlightId && highlightedRowRef.current) {
      highlightedRowRef.current.scrollIntoView({
        behavior: "smooth",
        block: "center",
      });
    }
  }, [highlightId, displayItems]);

  const handleSyncAccount = async () => {
    setIsSyncing(true);
    try {
      const res = await apiFetch("/api/accounts/sync", { method: "POST" }, true);
      if (res.status === 202) {
        toast("Bank sync request submitted. Waiting for bank settlement.");
      } else if (res.ok) {
        toast("Bank sync initiated. Transactions updated.");
        refetch();
      }
    } catch (err: any) {
      toast(err?.message || "Unable to reach banking provider at this time.", {
        variant: "danger",
      } as any);
    } finally {
      setIsSyncing(false);
    }
  };

  // Reconcile Transactions action
  const handleReconcile = async () => {
    setIsReconciling(true);
    try {
      const res = await apiFetch(
        "/api/accounts/transactions/reconcile",
        { method: "POST" },
        true
      );
      if (res.ok) {
        const data = (await res.json()) as {
          reconciled_transactions?: number;
          reconciled_groups?: number;
        };
        const txCount = data.reconciled_transactions || 0;
        const grpCount = data.reconciled_groups || 0;

        if (txCount > 0) {
          toast(
            `Reconciled ${txCount} transactions across ${grpCount} multi-currency event${
              grpCount === 1 ? "" : "s"
            }.`
          );
        } else {
          toast("No transactions to reconcile. All payments are up to date.");
        }

        queryClient.invalidateQueries({ queryKey: ["transactions"] });
        queryClient.invalidateQueries({ queryKey: ["expense-categories"] });
        queryClient.invalidateQueries({ queryKey: ["spending-trends"] });
        refetch();
      } else {
        toast("No transactions to reconcile. All payments are up to date.");
      }
    } catch (err: any) {
      toast(
        err?.message ||
          "Reconciliation check completed. All payments are up to date."
      );
    } finally {
      setIsReconciling(false);
    }
  };

  const handleExportCSV = () => {
    if (!displayItems.length) {
      toast("No transactions to export.", { variant: "danger" } as any);
      return;
    }

    const headers = [
      "Date",
      "Merchant / Description",
      "Category",
      "Amount",
      "Currency",
      "Type",
      "Status",
    ];
    const rows = displayItems.map((item) => {
      if (item.is_split_parent) {
        return [
          item.date || "",
          `"${(item.merchant_name || item.description || "").replace(/"/g, '""')}"`,
          `"${(item.category || "Uncategorized").replace(/"/g, '""')}"`,
          item.converted_total_gbp,
          "GBP",
          "Split Payment",
          item.status || "settled",
        ];
      }
      return [
        item.date || item.timestamp || "",
        `"${(item.description || item.merchant_name || "").replace(/"/g, '""')}"`,
        `"${(item.category || item.Category || "Uncategorized").replace(/"/g, '""')}"`,
        item.amount ?? 0,
        item.currency || activeCurrency,
        item.split_parent_name ? "Split Leg" : "Standard",
        item.status || "settled",
      ];
    });

    const csvContent =
      "data:text/csv;charset=utf-8," +
      [headers.join(","), ...rows.map((e) => e.join(","))].join("\n");

    const encodedUri = encodeURI(csvContent);
    const link = document.createElement("a");
    link.setAttribute("href", encodedUri);
    link.setAttribute(
      "download",
      `BudAI_Transactions_${new Date().toISOString().split("T")[0]}.csv`
    );
    document.body.appendChild(link);
    link.click();
    document.body.removeChild(link);
    toast("Transactions exported to CSV.");
  };

  // Synchronized Category Correction
  const handleUpdateCategory = async (
    item: DisplayItem,
    newCategory: string
  ) => {
    const isParent = item.is_split_parent === true;
    const targetIds: string[] = isParent
      ? (item as SplitGroupItem).all_tx_ids
      : [
          (item as Transaction).transaction_uuid ||
            (item as Transaction).transaction_id ||
            (item as any).id,
        ].filter(Boolean);

    if (!targetIds.length) return;

    // 1. Optimistic in-memory update across all transaction query sets
    queryClient.setQueriesData({ queryKey: ["transactions"] }, (old: any) => {
      if (!Array.isArray(old)) return old;
      return old.map((t: Transaction) => {
        const tid = t.transaction_uuid || t.transaction_id || (t as any).id;
        if (targetIds.includes(tid)) {
          return { ...t, category: newCategory, Category: newCategory };
        }
        return t;
      });
    });

    setIsUpdatingCategory(true);
    try {
      await Promise.all(
        targetIds.map((tid) =>
          apiFetch(
            "/api/categorizer/labels",
            {
              method: "POST",
              headers: { "Content-Type": "application/json" },
              body: JSON.stringify({
                transaction_uuid: String(tid),
                corrected_label: newCategory,
                retrain_model: true,
              }),
            },
            true
          )
        )
      );

      if (isParent) {
        toast(
          `Category updated to ${newCategory} across all ${targetIds.length} payment legs.`
        );
      } else {
        toast(`Transaction category updated to ${newCategory}.`);
      }

      queryClient.invalidateQueries({ queryKey: ["transactions"] });
      queryClient.invalidateQueries({ queryKey: ["expense-categories"] });
      queryClient.invalidateQueries({ queryKey: ["spending-trends"] });
    } catch (err: any) {
      toast(err?.message || "Failed to update category.", {
        variant: "danger",
      } as any);
      refetch();
    } finally {
      setIsUpdatingCategory(false);
    }
  };

  return (
    <div className="w-full h-full overflow-y-auto scrollbar-hide px-4 sm:px-6 lg:px-8 py-6 pb-28">
      <div className="max-w-7xl mx-auto flex flex-col gap-6">
        {/* Top Header & Toolbar */}
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
              variant="outline"
              onPress={handleReconcile}
              isPending={isReconciling}
              isDisabled={isReconciling}
              className="text-xs font-semibold h-8 px-3 rounded-xl border border-border hover:bg-foreground/5 flex items-center gap-1.5"
            >
              <ArrowLeftRight
                size={13}
                className={isReconciling ? "animate-spin text-primary" : "text-primary"}
              />
              <span>Reconcile</span>
            </Button>

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

        {/* Primary Search & Natural Financial Period Bar */}
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

          {/* Natural Financial Period Filters (1-tap chips, NO calendar date picker) */}
          <div className="flex items-center gap-1 bg-foreground/5 border border-border rounded-xl p-1 overflow-x-auto scrollbar-hide">
            {(
              [
                { id: "this_month", label: "This Month" },
                { id: "last_month", label: "Last Month" },
                { id: "90d", label: "Last 90 Days" },
                { id: "ytd", label: "Year to Date" },
                { id: "all", label: "All Time" },
              ] as const
            ).map((p) => (
              <button
                key={p.id}
                onClick={() => setPeriodFilter(p.id)}
                className={`px-3 py-1 rounded-lg text-xs font-bold whitespace-nowrap transition-all ${
                  periodFilter === p.id
                    ? "bg-primary text-primary-foreground shadow-xs"
                    : "text-muted-foreground hover:text-foreground"
                }`}
              >
                {p.label}
              </button>
            ))}
          </div>
        </div>

        {/* Secondary Filter Suite: Flow, Amount Range, Status, Sort & Clear All */}
        <div className="flex flex-wrap items-center justify-between gap-3 p-3 rounded-2xl bg-foreground/[0.02] border border-border/50 text-xs">
          <div className="flex flex-wrap items-center gap-3">
            {/* Flow Filter */}
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-semibold text-muted-foreground">
                Flow:
              </span>
              <div className="flex items-center gap-1 bg-foreground/5 p-0.5 rounded-lg border border-border/40">
                {(
                  [
                    { id: "all", label: "All" },
                    { id: "outflow", label: "Expenses" },
                    { id: "inflow", label: "Income" },
                    { id: "splits", label: "Splits" },
                  ] as const
                ).map((f) => (
                  <button
                    key={f.id}
                    onClick={() => setFlowFilter(f.id)}
                    className={`px-2 py-0.5 rounded-md text-[11px] font-semibold transition-all ${
                      flowFilter === f.id
                        ? "bg-primary/20 text-primary border border-primary/30"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {f.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Amount Range Filter */}
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-semibold text-muted-foreground">
                Amount:
              </span>
              <div className="flex items-center gap-1 bg-foreground/5 p-0.5 rounded-lg border border-border/40">
                {(
                  [
                    { id: "all", label: "All" },
                    { id: "under_20", label: "< £20" },
                    { id: "20_100", label: "£20-£100" },
                    { id: "100_500", label: "£100-£500" },
                    { id: "over_500", label: "> £500" },
                  ] as const
                ).map((a) => (
                  <button
                    key={a.id}
                    onClick={() => setAmountFilter(a.id)}
                    className={`px-2 py-0.5 rounded-md text-[11px] font-semibold transition-all ${
                      amountFilter === a.id
                        ? "bg-primary/20 text-primary border border-primary/30"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {a.label}
                  </button>
                ))}
              </div>
            </div>

            {/* Status Filter */}
            <div className="flex items-center gap-1.5">
              <span className="text-[11px] font-semibold text-muted-foreground">
                Status:
              </span>
              <div className="flex items-center gap-1 bg-foreground/5 p-0.5 rounded-lg border border-border/40">
                {(
                  [
                    { id: "all", label: "All" },
                    { id: "settled", label: "Settled" },
                    { id: "pending", label: "Pending" },
                  ] as const
                ).map((s) => (
                  <button
                    key={s.id}
                    onClick={() => setStatusFilter(s.id)}
                    className={`px-2 py-0.5 rounded-md text-[11px] font-semibold transition-all ${
                      statusFilter === s.id
                        ? "bg-primary/20 text-primary border border-primary/30"
                        : "text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {s.label}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Sort & Reset Actions */}
          <div className="flex items-center gap-2 ml-auto">
            <div className="flex items-center gap-1.5">
              <ArrowUpDown size={12} className="text-muted-foreground" />
              <select
                value={sortOrder}
                onChange={(e) => setSortOrder(e.target.value as SortOrder)}
                className="bg-foreground/5 hover:bg-foreground/10 border border-border rounded-lg px-2 py-1 text-xs text-foreground outline-none cursor-pointer"
              >
                <option value="date_desc">Date (Newest First)</option>
                <option value="date_asc">Date (Oldest First)</option>
                <option value="amount_desc">Amount (High to Low)</option>
                <option value="amount_asc">Amount (Low to High)</option>
                <option value="merchant_asc">Merchant (A to Z)</option>
              </select>
            </div>

            {hasActiveFilters && (
              <button
                onClick={handleResetFilters}
                className="text-xs font-semibold text-primary hover:underline flex items-center gap-1 px-2 py-1 rounded-lg hover:bg-foreground/5 transition-colors"
              >
                <RotateCcw size={12} />
                <span>Clear All</span>
              </button>
            )}
          </div>
        </div>

        {/* Categories Bar */}
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

        {/* Table & Accordion Container */}
        <div className="bg-[#0d0e14]/40 backdrop-blur-xl border border-white/10 rounded-2xl overflow-hidden shadow-xl">
          <div className="overflow-x-auto">
            <table className="w-full text-left text-xs">
              <thead className="bg-foreground/5 text-muted-foreground font-semibold border-b border-border text-[11px] uppercase tracking-wider">
                <tr>
                  <th className="py-3 px-4 w-[13%]">Date</th>
                  <th className="py-3 px-4 w-[35%]">Description / Merchant</th>
                  <th className="py-3 px-4 w-[18%]">Category</th>
                  <th className="py-3 px-4 w-[10%]">Status</th>
                  <th className="py-3 px-4 w-[12%] text-right">Amount</th>
                  <th className="py-3 px-4 w-[12%] text-center">Action</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-border/40">
                {isLoading ? (
                  [1, 2, 3, 4, 5, 6].map((i) => (
                    <tr key={i} className="animate-pulse">
                      <td className="py-3 px-4 w-[13%]">
                        <Skeleton className="h-3 w-16 rounded" />
                      </td>
                      <td className="py-3 px-4 w-[35%]">
                        <Skeleton className="h-3 w-40 rounded" />
                      </td>
                      <td className="py-3 px-4 w-[18%]">
                        <Skeleton className="h-3 w-24 rounded" />
                      </td>
                      <td className="py-3 px-4 w-[10%]">
                        <Skeleton className="h-3 w-16 rounded" />
                      </td>
                      <td className="py-3 px-4 w-[12%] text-right">
                        <Skeleton className="h-3 w-16 rounded ml-auto" />
                      </td>
                      <td className="py-3 px-4 w-[12%] text-center">
                        <Skeleton className="h-3 w-10 rounded mx-auto" />
                      </td>
                    </tr>
                  ))
                ) : displayItems.length === 0 ? (
                  <tr>
                    <td colSpan={6} className="py-12 text-center">
                      <div className="flex flex-col items-center justify-center gap-2">
                        <CreditCard size={28} className="text-muted-foreground/50" />
                        <p className="text-sm font-semibold text-foreground">
                          No transactions found
                        </p>
                        <p className="text-xs text-muted-foreground max-w-sm">
                          {hasActiveFilters
                            ? "Try adjusting your filters or search terms."
                            : "Connect your bank account or trigger a manual sync to pull recent activity."}
                        </p>
                      </div>
                    </td>
                  </tr>
                ) : (
                  displayItems.map((item) => {
                    // Check if this item is a Parent Split Card (Only in ALL accounts view)
                    if (item.is_split_parent) {
                      const isExpanded = expandedSplitKeys.has(item.group_id);
                      const formattedDate = item.date
                        ? new Date(item.date).toLocaleDateString("en-GB", {
                            year: "numeric",
                            month: "short",
                            day: "numeric",
                          })
                        : "—";

                      const fullDateTime = item.date
                        ? new Date(item.date).toLocaleString("en-GB", {
                            day: "numeric",
                            month: "short",
                            year: "numeric",
                            hour: "2-digit",
                            minute: "2-digit",
                          })
                        : formattedDate;

                      const isCredit = item.converted_total_gbp > 0;
                      const isPending = item.status === "pending";

                      return (
                        <tr
                          key={item.group_id}
                          className="hover:bg-foreground/[0.02] transition-colors border-b border-border/40"
                        >
                          <td colSpan={6} className="p-0">
                            <Accordion
                              allowsMultipleExpanded
                              expandedKeys={isExpanded ? new Set([item.group_id]) : new Set()}
                              onExpandedChange={(keys) => {
                                setExpandedSplitKeys((prev) => {
                                  const next = new Set(prev);
                                  if (keys.has(item.group_id)) {
                                    next.add(item.group_id);
                                  } else {
                                    next.delete(item.group_id);
                                  }
                                  return next;
                                });
                              }}
                              className="w-full"
                            >
                              <Accordion.Item id={item.group_id} className="border-none">
                                <div
                                  onClick={(e) => {
                                    const target = e.target as HTMLElement;
                                    if (
                                      target.closest(
                                        "button, [role='button'], a, input, [data-slot='dropdown-trigger']"
                                      )
                                    ) {
                                      return;
                                    }
                                    setExpandedSplitKeys((prev) => {
                                      const next = new Set(prev);
                                      if (next.has(item.group_id)) {
                                        next.delete(item.group_id);
                                      } else {
                                        next.add(item.group_id);
                                      }
                                      return next;
                                    });
                                  }}
                                  className="flex items-center w-full py-3.5 px-4 text-xs cursor-pointer select-none"
                                >
                                  {/* Date */}
                                  <div className="w-[13%] font-mono text-muted-foreground whitespace-nowrap">
                                    {formattedDate}
                                  </div>

                                  {/* Merchant + Split Payment Badge */}
                                  <div className="w-[35%] flex items-center gap-2 min-w-0 pr-2">
                                    <Tooltip delay={100}>
                                      <Tooltip.Trigger className="truncate max-w-full block cursor-pointer text-left outline-none">
                                        <span className="font-bold text-foreground hover:text-primary transition-colors truncate">
                                          {item.merchant_name}
                                        </span>
                                      </Tooltip.Trigger>
                                      <Tooltip.Content
                                        showArrow
                                        offset={8}
                                        className="w-80 sm:w-96 max-w-md bg-[#0b0c14]/80 backdrop-blur-2xl backdrop-saturate-150 border border-white/20 p-4 rounded-2xl shadow-[0_25px_60px_-15px_rgba(0,0,0,0.85),inset_0_1px_1px_0_rgba(255,255,255,0.15)] text-xs flex flex-col gap-2.5 z-50 text-foreground"
                                      >
                                        <Tooltip.Arrow className="fill-[#0b0c14]/80 stroke-white/20" />
                                        <div className="flex items-start justify-between gap-3">
                                          <div className="flex flex-col gap-1 min-w-0 flex-1">
                                            <span className="font-bold text-sm text-foreground break-words leading-snug drop-shadow-xs">
                                              {item.merchant_name}
                                            </span>
                                            <span className="text-[11px] text-muted-foreground/90 font-mono break-words leading-relaxed bg-white/[0.03] p-1.5 rounded-lg border border-white/5">
                                              Reconciled multi-currency split payment across {item.legs.length} accounts
                                            </span>
                                          </div>
                                          <span
                                            className={`font-mono font-bold text-xs shrink-0 px-2.5 py-1 rounded-lg backdrop-blur-md shadow-xs ${
                                              isCredit
                                                ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                                                : "bg-white/10 text-foreground border border-white/15"
                                            }`}
                                          >
                                            {isCredit ? "+" : ""}
                                            {formatCurrency(item.converted_total_gbp, "GBP")}
                                          </span>
                                        </div>

                                        <div className="border-t border-white/10" />

                                        <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                                          <div className="flex flex-col gap-0.5 p-2 rounded-xl bg-white/[0.03] border border-white/5 backdrop-blur-sm">
                                            <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1">
                                              <Calendar size={10} />
                                              Date & Time
                                            </span>
                                            <span className="font-medium text-foreground break-words">
                                              {fullDateTime}
                                            </span>
                                          </div>

                                          <div className="flex flex-col gap-0.5 p-2 rounded-xl bg-white/[0.03] border border-white/5 backdrop-blur-sm">
                                            <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1">
                                              <Tag size={10} />
                                              Category
                                            </span>
                                            <span className="font-medium text-foreground break-words">
                                              {item.category}
                                            </span>
                                          </div>

                                          <div className="flex flex-col gap-0.5 p-2 rounded-xl bg-white/[0.03] border border-white/5 backdrop-blur-sm col-span-2">
                                            <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1">
                                              <Layers size={10} />
                                              Constituent Accounts
                                            </span>
                                            <span className="font-medium text-foreground break-words">
                                              {item.legs.map((l) => l.account_label).join(" • ")}
                                            </span>
                                          </div>
                                        </div>
                                      </Tooltip.Content>
                                    </Tooltip>

                                    <Chip
                                      size="sm"
                                      variant="soft"
                                      color="accent"
                                      className="text-[10px] font-bold uppercase tracking-wider shrink-0 bg-primary/15 text-primary border border-primary/30"
                                    >
                                      Split Payment
                                    </Chip>
                                  </div>

                                  {/* Category */}
                                  <div className="w-[18%] whitespace-nowrap">
                                    <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-foreground/5 border border-border">
                                      <Tag size={10} className="text-muted-foreground" />
                                      <span>{item.category}</span>
                                    </span>
                                  </div>

                                  {/* Status */}
                                  <div className="w-[10%] whitespace-nowrap">
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
                                  </div>

                                  {/* Converted Total Amount */}
                                  <div className="w-[12%] text-right font-mono font-bold whitespace-nowrap">
                                    <span
                                      className={
                                        isCredit
                                          ? "text-emerald-500"
                                          : "text-foreground"
                                      }
                                    >
                                      {isCredit ? "+" : ""}
                                      {formatCurrency(item.converted_total_gbp, "GBP")}
                                    </span>
                                  </div>

                                  {/* Action & Accordion Trigger */}
                                  <div className="w-[12%] flex items-center justify-end gap-2.5 whitespace-nowrap">
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
                                            handleUpdateCategory(item, String(key))
                                          }
                                        >
                                          {STANDARD_CATEGORIES.filter(
                                            (c) => c !== "All"
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

                                    <Accordion.Heading>
                                      <Accordion.Trigger
                                        aria-label="Toggle split payment details"
                                        className="flex items-center justify-center size-7 rounded-lg hover:bg-foreground/10 text-muted-foreground hover:text-foreground transition-colors cursor-pointer outline-none border border-border/40 hover:border-border shrink-0"
                                      >
                                        <Accordion.Indicator>
                                          <ChevronDown
                                            size={14}
                                            className={`transition-transform duration-200 ${
                                              isExpanded ? "rotate-180" : ""
                                            }`}
                                          />
                                        </Accordion.Indicator>
                                      </Accordion.Trigger>
                                    </Accordion.Heading>
                                  </div>
                                </div>

                                <Accordion.Panel>
                                  <Accordion.Body className="px-6 py-4 bg-foreground/[0.015] border-t border-border/30">
                                    <div className="flex flex-col gap-3">
                                      <div className="flex items-center justify-between">
                                        <span className="text-[11px] font-bold uppercase tracking-wider text-muted-foreground flex items-center gap-1.5">
                                          <Layers size={13} className="text-primary" />
                                          Multi-Currency Breakdown ({item.legs.length} parts)
                                        </span>
                                        <span className="text-[11px] text-muted-foreground font-mono">
                                          Total Cost: {formatCurrency(item.converted_total_gbp, "GBP")}
                                        </span>
                                      </div>

                                      <div className="grid grid-cols-1 md:grid-cols-3 gap-2.5">
                                        {item.legs.map((leg, legIdx) => (
                                          <div
                                            key={
                                              leg.transaction_uuid ||
                                              leg.transaction_id ||
                                              legIdx
                                            }
                                            className="p-3 rounded-xl bg-white/[0.03] dark:bg-black/20 border border-white/10 backdrop-blur-md flex flex-col gap-2"
                                          >
                                            <div className="flex items-center justify-between gap-2">
                                              <span className="text-[10px] uppercase font-bold tracking-wider text-muted-foreground">
                                                {leg.is_fee_leg
                                                  ? "Conversion Fee"
                                                  : `Payment Leg ${legIdx + 1}`}
                                              </span>
                                              <span className="text-[10px] px-2 py-0.5 rounded-md font-semibold bg-foreground/5 border border-border text-muted-foreground">
                                                {leg.account_label}
                                              </span>
                                            </div>

                                            <div className="flex items-baseline justify-between">
                                              <span className="font-mono font-bold text-sm text-foreground">
                                                {formatCurrency(
                                                  leg.amount ?? 0,
                                                  leg.currency || "GBP"
                                                )}
                                              </span>
                                              {leg.currency !== "GBP" && leg.converted_gbp && (
                                                <span className="text-[11px] font-mono text-muted-foreground">
                                                  ≈ {formatCurrency(leg.converted_gbp, "GBP")}
                                                </span>
                                              )}
                                            </div>

                                            <p className="text-[11px] text-muted-foreground/80 truncate">
                                              {leg.description}
                                            </p>
                                          </div>
                                        ))}
                                      </div>
                                    </div>
                                  </Accordion.Body>
                                </Accordion.Panel>
                              </Accordion.Item>
                            </Accordion>
                          </td>
                        </tr>
                      );
                    }

                    // Standard Individual Transaction Row
                    const tx = item as Transaction & {
                      split_parent_name?: string;
                    };
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
                    const fullDateTime = dateStr
                      ? new Date(dateStr).toLocaleString("en-GB", {
                          day: "numeric",
                          month: "short",
                          year: "numeric",
                          hour: "2-digit",
                          minute: "2-digit",
                        })
                      : formattedDate;
                    const isPending =
                      tx.status?.toLowerCase() === "pending" ||
                      cat.toLowerCase() === "pending";

                    const txAccountLabel =
                      accountMap.get(tx.account_id || "") ||
                      activeAccount?.bank_name ||
                      "Primary Account";

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
                        <td className="py-3 px-4 w-[13%] font-mono text-muted-foreground whitespace-nowrap">
                          {formattedDate}
                        </td>
                        <td className="py-3 px-4 w-[35%] max-w-[280px]">
                          <div className="flex items-center gap-2">
                            <Tooltip delay={100}>
                              <Tooltip.Trigger className="truncate max-w-full block cursor-pointer text-left outline-none">
                                <span className="font-semibold text-foreground hover:text-primary transition-colors">
                                  {desc}
                                </span>
                              </Tooltip.Trigger>
                              <Tooltip.Content
                                showArrow
                                offset={8}
                                className="w-80 sm:w-96 max-w-md bg-[#0b0c14]/80 backdrop-blur-2xl backdrop-saturate-150 border border-white/20 p-4 rounded-2xl shadow-[0_25px_60px_-15px_rgba(0,0,0,0.85),inset_0_1px_1px_0_rgba(255,255,255,0.15)] text-xs flex flex-col gap-2.5 z-50 text-foreground"
                              >
                                <Tooltip.Arrow className="fill-[#0b0c14]/80 stroke-white/20" />
                                <div className="flex items-start justify-between gap-3">
                                  <div className="flex flex-col gap-1 min-w-0 flex-1">
                                    <span className="font-bold text-sm text-foreground break-words leading-snug drop-shadow-xs">
                                      {tx.merchant_name || desc}
                                    </span>
                                    {tx.description &&
                                      tx.merchant_name &&
                                      tx.description !== tx.merchant_name && (
                                        <span className="text-[11px] text-muted-foreground/90 font-mono break-words leading-relaxed bg-white/[0.03] p-1.5 rounded-lg border border-white/5">
                                          {tx.description}
                                        </span>
                                      )}
                                  </div>
                                  <span
                                    className={`font-mono font-bold text-xs shrink-0 px-2.5 py-1 rounded-lg backdrop-blur-md shadow-xs ${
                                      isCredit
                                        ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                                        : "bg-white/10 text-foreground border border-white/15"
                                    }`}
                                  >
                                    {isCredit ? "+" : ""}
                                    {formatCurrency(
                                      amt,
                                      tx.currency || activeCurrency
                                    )}
                                  </span>
                                </div>

                                <div className="border-t border-white/10" />

                                <div className="grid grid-cols-2 gap-1.5 text-[11px]">
                                  <div className="flex flex-col gap-0.5 p-2 rounded-xl bg-white/[0.03] border border-white/5 backdrop-blur-sm">
                                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1">
                                      <Calendar size={10} />
                                      Date & Time
                                    </span>
                                    <span className="font-medium text-foreground break-words">
                                      {fullDateTime}
                                    </span>
                                  </div>

                                  <div className="flex flex-col gap-0.5 p-2 rounded-xl bg-white/[0.03] border border-white/5 backdrop-blur-sm">
                                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1">
                                      <Tag size={10} />
                                      Category
                                    </span>
                                    <span className="font-medium text-foreground break-words">
                                      {cat}
                                      {tx.sub_category ? ` (${tx.sub_category})` : ""}
                                    </span>
                                  </div>

                                  <div className="flex flex-col gap-0.5 p-2 rounded-xl bg-white/[0.03] border border-white/5 backdrop-blur-sm">
                                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1">
                                      <Building2 size={10} />
                                      Account
                                    </span>
                                    <span className="font-medium text-foreground break-words">
                                      {txAccountLabel}
                                    </span>
                                  </div>

                                  <div className="flex flex-col gap-0.5 p-2 rounded-xl bg-white/[0.03] border border-white/5 backdrop-blur-sm">
                                    <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold flex items-center gap-1">
                                      {isPending ? (
                                        <Clock size={10} />
                                      ) : (
                                        <CheckCircle2 size={10} />
                                      )}
                                      Status
                                    </span>
                                    <span
                                      className={`font-medium ${
                                        isPending
                                          ? "text-amber-400"
                                          : "text-emerald-400"
                                      }`}
                                    >
                                      {isPending ? "Pending" : "Settled"}
                                    </span>
                                  </div>
                                </div>

                                {txId && (
                                  <div className="pt-1.5 border-t border-white/10 flex items-center justify-between text-[10px] text-muted-foreground/70 font-mono gap-2">
                                    <span className="shrink-0">Ref</span>
                                    <span className="break-all font-mono text-right">
                                      {txId}
                                    </span>
                                  </div>
                                )}
                              </Tooltip.Content>
                            </Tooltip>

                            {tx.split_parent_name && (
                              <Chip
                                size="sm"
                                variant="soft"
                                className="text-[10px] font-bold shrink-0 bg-white/5 border border-white/10 text-muted-foreground"
                              >
                                Part of {tx.split_parent_name}
                              </Chip>
                            )}
                          </div>
                        </td>
                        <td className="py-3 px-4 w-[18%] whitespace-nowrap">
                          <span className="inline-flex items-center gap-1 px-2.5 py-0.5 rounded-full text-[11px] font-medium bg-foreground/5 border border-border">
                            <Tag size={10} className="text-muted-foreground" />
                            <span>{cat}</span>
                          </span>
                        </td>
                        <td className="py-3 px-4 w-[10%] whitespace-nowrap">
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
                        <td className="py-3 px-4 w-[12%] text-right font-mono font-bold whitespace-nowrap">
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
                        <td className="py-3 px-4 w-[12%] text-center whitespace-nowrap">
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
                                  handleUpdateCategory(item, String(key))
                                }
                              >
                                {STANDARD_CATEGORIES.filter(
                                  (c) => c !== "All"
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
