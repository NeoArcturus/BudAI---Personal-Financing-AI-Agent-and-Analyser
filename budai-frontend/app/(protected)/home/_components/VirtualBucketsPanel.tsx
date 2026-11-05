"use client";

import React, { useState, useMemo } from "react";
import { Button, Dropdown, Skeleton } from "@heroui/react";
import {
  Plus,
  ArrowLeftRight,
  MoreVertical,
  Shield,
  Receipt,
  PiggyBank,
  Target,
  CreditCard,
  Edit2,
  Trash2,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { formatCurrency, getCurrencySymbol } from "@/lib/currency";
import { useBudAI } from "@/app/context/AppContext";
import { CreateBucketModal } from "@/app/(protected)/_components/widgets/CreateBucketModal";
import { EditBucketModal } from "@/app/(protected)/_components/widgets/EditBucketModal";
import { DeleteBucketModal } from "@/app/(protected)/_components/widgets/DeleteBucketModal";
import { TransferMoneyModal } from "@/app/(protected)/_components/widgets/TransferMoneyModal";

interface VirtualBucketsPanelProps {
  currency?: string;
  initialData?: any[];
}

export const VirtualBucketsPanel = ({
  currency = "GBP",
  initialData = [],
}: VirtualBucketsPanelProps) => {
  const { accounts, activeAccountId } = useBudAI();
  const [isCreateOpen, setIsCreateOpen] = useState(false);
  const [isTransferOpen, setIsTransferOpen] = useState(false);
  const [isEditOpen, setIsEditOpen] = useState(false);
  const [isDeleteOpen, setIsDeleteOpen] = useState(false);
  const [selectedBucket, setSelectedBucket] = useState<any>(null);
  const [transferTargetId, setTransferTargetId] = useState<string | undefined>(
    undefined,
  );

  const activeAccountCurrency = useMemo(() => {
    if (activeAccountId && activeAccountId !== "ALL") {
      const acc = accounts.find((a) => a.account_id === activeAccountId);
      if (acc?.currency) return acc.currency;
    }
    return accounts[0]?.currency || currency;
  }, [accounts, activeAccountId, currency]);

  const { data: bucketData, isLoading } = useQuery({
    queryKey: ["buckets", activeAccountCurrency],
    queryFn: async () => {
      const res = await apiFetch(
        `/api/buckets/?base_currency=${encodeURIComponent(activeAccountCurrency)}`
      );
      if (res.ok) {
        const json = await res.json();
        if (Array.isArray(json)) return { buckets: json };
        return json as {
          buckets?: any[];
          default_bucket_id?: string;
          default_bucket_balance?: number;
          reconciliation?: {
            total_physical_reconciled?: number;
            total_virtual_reconciled?: number;
            discrepancy?: number;
            is_balanced?: boolean;
            by_currency?: Record<
              string,
              { nominal: number; converted: number; rate: number }
            >;
          };
        };
      }
      return { buckets: initialData };
    },
    initialData: initialData.length > 0 ? { buckets: initialData } : undefined,
    refetchInterval: 15000,
  });

  const buckets = useMemo(() => bucketData?.buckets || [], [bucketData]);

  const foreignCurrencies = useMemo(() => {
    const list: { currency: string; balance: number }[] = [];
    const base = (activeAccountCurrency || "GBP").toUpperCase();
    const currMap: Record<string, number> = {};

    accounts.forEach((acc) => {
      const c = (acc.currency || base).toUpperCase();
      if (c !== base) {
        currMap[c] = (currMap[c] || 0) + (acc.balance || 0);
      }
    });

    Object.entries(currMap).forEach(([curr, bal]) => {
      list.push({ currency: curr, balance: bal });
    });
    return list;
  }, [accounts, activeAccountCurrency]);

  // Query live FX rates for any non-base currency in linked accounts
  const { data: fxRates = {} } = useQuery({
    queryKey: [
      "fx-rates",
      activeAccountCurrency,
      foreignCurrencies.map((f) => f.currency).sort().join(","),
    ],
    queryFn: async () => {
      if (foreignCurrencies.length === 0) return {};
      const rates: Record<string, number> = {};
      await Promise.all(
        foreignCurrencies.map(async ({ currency: curr }) => {
          try {
            const res = await apiFetch(
              `/api/market/fx?from_currency=${encodeURIComponent(
                curr
              )}&to_currency=${encodeURIComponent(
                activeAccountCurrency
              )}&amount=1`
            );
            if (res.ok) {
              const json = (await res.json()) as {
                data?: { rate?: number };
              } | null;
              const rate = json?.data?.rate;
              if (typeof rate === "number" && rate > 0) {
                rates[curr] = rate;
              }
            }
          } catch {
            // Keep rate retrieval resilient
          }
        })
      );
      return rates;
    },
    enabled: foreignCurrencies.length > 0,
    staleTime: 60000,
  });

  const {
    totalPhysical,
    totalBuckets,
    diff,
    isBalanced,
    hasForeignAccounts,
    foreignSummary,
  } = useMemo(() => {
    // 1. Authoritative Backend Reconciliation payload (when server feature is active)
    if (
      bucketData?.reconciliation &&
      typeof bucketData.reconciliation.total_physical_reconciled === "number"
    ) {
      const r = bucketData.reconciliation;
      const physical = r.total_physical_reconciled ?? 0;
      const inBuckets =
        r.total_virtual_reconciled ??
        buckets.reduce(
          (sum: number, b: any) => sum + (b.cached_balance || 0),
          0
        );
      const d = r.discrepancy ?? (physical - inBuckets);
      return {
        totalPhysical: physical,
        totalBuckets: inBuckets,
        diff: d,
        isBalanced: r.is_balanced ?? Math.abs(d) < 0.01,
        hasForeignAccounts: foreignCurrencies.length > 0,
        foreignSummary: foreignCurrencies
          .map((f) => formatCurrency(f.balance, f.currency))
          .join(", "),
      };
    }

    // 2. Client-side FX-converted reconciliation
    const relevantAccounts =
      !activeAccountId || activeAccountId === "ALL"
        ? accounts
        : accounts.filter((a) => a.account_id === activeAccountId);

    const baseUpper = (activeAccountCurrency || "GBP").toUpperCase();

    let physical = 0;
    relevantAccounts.forEach((acc) => {
      const accCurr = (acc.currency || baseUpper).toUpperCase();
      const bal = acc.balance || 0;
      if (accCurr === baseUpper) {
        physical += bal;
      } else {
        const rate = fxRates[accCurr];
        if (typeof rate === "number" && rate > 0) {
          physical += bal * rate;
        }
        // Exclude unweighted nominal foreign balances until conversion rate arrives
      }
    });

    const inBuckets = buckets.reduce(
      (sum: number, b: any) => sum + (b.cached_balance || 0),
      0
    );
    const d = physical - inBuckets;
    return {
      totalPhysical: physical,
      totalBuckets: inBuckets,
      diff: d,
      isBalanced: Math.abs(d) < 0.01,
      hasForeignAccounts: foreignCurrencies.length > 0,
      foreignSummary: foreignCurrencies
        .map((f) => formatCurrency(f.balance, f.currency))
        .join(", "),
    };
  }, [
    bucketData,
    accounts,
    activeAccountId,
    activeAccountCurrency,
    buckets,
    foreignCurrencies,
    fxRates,
  ]);

  const getBucketTypeMeta = (type: string) => {
    switch (type) {
      case "DEFAULT":
        return {
          label: "Daily Spending",
          icon: Shield,
          color: "text-blue-500",
          bg: "bg-blue-500/10 border-blue-500/20",
          barColor: "bg-blue-500",
        };
      case "RECURRING":
        return {
          label: "Bills & Subscriptions",
          icon: Receipt,
          color: "text-amber-500",
          bg: "bg-amber-500/10 border-amber-500/20",
          barColor: "bg-amber-500",
        };
      case "ACCUMULATING":
        return {
          label: "Savings Reserve",
          icon: PiggyBank,
          color: "text-emerald-500",
          bg: "bg-emerald-500/10 border-emerald-500/20",
          barColor: "bg-emerald-500",
        };
      case "TARGET_DATE":
        return {
          label: "Goal",
          icon: Target,
          color: "text-indigo-500",
          bg: "bg-indigo-500/10 border-indigo-500/20",
          barColor: "bg-indigo-500",
        };
      case "LIABILITY":
        return {
          label: "Debt & Borrowing",
          icon: CreditCard,
          color: "text-red-500",
          bg: "bg-red-500/10 border-red-500/20",
          barColor: "bg-red-500",
        };
      default:
        return {
          label: "Custom",
          icon: Shield,
          color: "text-foreground",
          bg: "bg-foreground/5 border-border",
          barColor: "bg-primary",
        };
    }
  };

  const handleOpenTransferForBucket = (bucketId: string) => {
    setTransferTargetId(bucketId);
    setIsTransferOpen(true);
  };

  return (
    <div className="flex flex-col gap-4 w-full">
      <div className="flex items-center justify-between pb-1">
        <div>
          <h2 className="text-base font-bold text-foreground">Money Buckets</h2>
          <p className="text-xs text-muted-foreground mt-0.5">
            Allocated categories and reserves
          </p>
        </div>
        <div className="flex items-center gap-2">
          <Button
            size="sm"
            variant="ghost"
            onPress={() => {
              setTransferTargetId(undefined);
              setIsTransferOpen(true);
            }}
            className="h-8 px-3 text-xs font-semibold rounded-xl bg-white/5 hover:bg-white/10 border border-white/10 backdrop-blur-md flex items-center gap-1.5 transition-all"
          >
            <ArrowLeftRight size={13} />
            <span>Move</span>
          </Button>
          <Button
            size="sm"
            onPress={() => setIsCreateOpen(true)}
            className="h-8 px-3 text-xs font-bold rounded-xl bg-primary text-primary-foreground hover:opacity-90 shadow-sm flex items-center gap-1.5"
          >
            <Plus size={14} />
            <span>New</span>
          </Button>
        </div>
      </div>

      {/* Balance Match Check Card - Frosted Glass */}
      <div className="p-5 rounded-2xl bg-[#0d0e14]/40 backdrop-blur-xl border border-white/10 shadow-xl flex flex-col gap-3">
        <div className="flex items-center justify-between">
          <span className="text-xs font-semibold text-muted-foreground">
            Balance Match Check
          </span>
          <span
            className={`inline-flex items-center gap-1.5 px-3 py-1 rounded-full text-xs font-bold ${
              isBalanced
                ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border border-emerald-500/25"
                : diff > 0
                ? "bg-amber-500/15 text-amber-600 dark:text-amber-400 border border-amber-500/25"
                : "bg-red-500/15 text-red-600 dark:text-red-400 border border-red-500/25"
            }`}
          >
            {isBalanced ? (
              <CheckCircle2 size={13} />
            ) : (
              <AlertCircle size={13} />
            )}
            {isBalanced
              ? "All Balanced"
              : diff > 0
              ? `+${getCurrencySymbol(activeAccountCurrency)}${diff.toFixed(2)} Unassigned`
              : `-${getCurrencySymbol(activeAccountCurrency)}${Math.abs(diff).toFixed(2)} Overallocated`}
          </span>
        </div>
        <div className="flex items-baseline justify-between pt-2 border-t border-white/10 dark:border-white/5">
          <div>
            <div className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold">
              In Buckets
            </div>
            <div className="text-xl font-bold text-foreground font-mono mt-0.5">
              {formatCurrency(totalBuckets, activeAccountCurrency)}
            </div>
          </div>
          <div className="text-right">
            <div className="text-[11px] uppercase tracking-wider text-muted-foreground font-semibold">
              Bank Balance
            </div>
            <div className="text-base font-bold text-muted-foreground font-mono mt-0.5">
              {formatCurrency(totalPhysical, activeAccountCurrency)}
            </div>
            {hasForeignAccounts && (!activeAccountId || activeAccountId === "ALL") && (
              <div className="text-[10px] text-muted-foreground/80 mt-0.5">
                Includes FX converted {foreignSummary}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Buckets List */}
      <div className="flex flex-col gap-3">
        {isLoading ? (
          <div className="flex flex-col gap-3">
            {[1, 2, 3].map((i) => (
              <div
                key={i}
                className="p-4 rounded-2xl border border-white/10 bg-white/40 dark:bg-neutral-900/40 backdrop-blur-xl flex flex-col gap-2.5"
              >
                <div className="flex justify-between items-center">
                  <Skeleton className="h-4 w-32 rounded-md" />
                  <Skeleton className="h-4 w-20 rounded-md" />
                </div>
                <Skeleton className="h-2 w-full rounded-full" />
              </div>
            ))}
          </div>
        ) : buckets.length === 0 ? (
          <div className="p-8 text-center rounded-2xl border border-dashed border-white/15 bg-white/40 dark:bg-neutral-900/30 backdrop-blur-xl flex flex-col items-center gap-2.5">
            <Shield size={28} className="text-muted-foreground/60" />
            <p className="text-sm font-semibold text-foreground">
              No money buckets set up yet
            </p>
            <p className="text-xs text-muted-foreground max-w-xs">
              Create your first category or connect an account to organize your finances.
            </p>
            <Button
              size="sm"
              onPress={() => setIsCreateOpen(true)}
              className="mt-2 text-xs font-bold bg-primary text-primary-foreground rounded-xl px-4 py-2"
            >
              Add First Bucket
            </Button>
          </div>
        ) : (
          buckets.map((bucket: any) => {
            const bucketId =
              bucket.id || bucket.bucket_id || bucket.bucket_uuid;
            const bType = bucket.bucket_type || bucket.type || "DEFAULT";
            const meta = getBucketTypeMeta(bType);
            const Icon = meta.icon;
            const isDefault =
              bType === "DEFAULT" ||
              (typeof bucket.name === "string" &&
                bucket.name.toLowerCase().includes("unallocated"));
            const rawBalance = bucket.cached_balance || 0;
            const balance = isDefault
              ? (rawBalance > 0 ? rawBalance : Math.max(0, diff))
              : rawBalance;
            const target = bucket.target_amount || 0;
            const progress =
              target > 0 ? Math.min(100, Math.round((balance / target) * 100)) : 0;

            return (
              <div
                key={bucketId}
                className="group relative p-4 rounded-2xl bg-[#0d0e14]/40 backdrop-blur-xl border border-white/10 hover:border-white/20 transition-all shadow-sm flex flex-col gap-2.5"
              >
                <div className="flex items-start justify-between gap-2.5">
                  <div className="flex items-center gap-3 min-w-0">
                    <div
                      className={`w-9 h-9 rounded-xl flex items-center justify-center shrink-0 border shadow-xs ${meta.bg}`}
                    >
                      <Icon size={16} className={meta.color} />
                    </div>
                    <div className="min-w-0">
                      <div className="flex items-center gap-2">
                        <span className="text-sm font-bold text-foreground truncate">
                          {bucket.name || meta.label}
                        </span>
                        {isDefault && (
                          <span className="text-[10px] font-bold px-1.5 py-0.5 rounded-md bg-foreground/10 text-muted-foreground uppercase tracking-wider">
                            Main
                          </span>
                        )}
                      </div>
                      <span className="text-xs text-muted-foreground block truncate mt-0.5">
                        {meta.label}
                      </span>
                    </div>
                  </div>

                  <div className="flex items-center gap-1.5 shrink-0">
                    <span className="text-sm font-bold text-foreground font-mono">
                      {formatCurrency(balance, activeAccountCurrency)}
                    </span>
                    <Dropdown>
                      <Button
                        isIconOnly
                        aria-label="Bucket options"
                        variant="ghost"
                        className="w-7 h-7 min-w-0 rounded-lg hover:bg-foreground/5 flex items-center justify-center text-muted-foreground hover:text-foreground transition-colors cursor-pointer shadow-none"
                      >
                        <MoreVertical size={14} />
                      </Button>
                      <Dropdown.Popover className="bg-popover/95 backdrop-blur-2xl border border-border shadow-2xl rounded-xl w-48 p-1.5 z-50">
                        <Dropdown.Menu
                          onAction={(key) => {
                            if (key === "transfer")
                              handleOpenTransferForBucket(bucketId);
                            else if (key === "edit") {
                              setSelectedBucket(bucket);
                              setIsEditOpen(true);
                            } else if (key === "delete") {
                              setSelectedBucket(bucket);
                              setIsDeleteOpen(true);
                            }
                          }}
                        >
                          <Dropdown.Item
                            id="transfer"
                            textValue="Move Money"
                            className="px-3 py-2 rounded-lg text-xs hover:bg-foreground/5 cursor-pointer flex items-center gap-2"
                          >
                            <ArrowLeftRight size={13} />
                            <span>Move Money</span>
                          </Dropdown.Item>
                          {!isDefault && (
                            <Dropdown.Item
                              id="edit"
                              textValue="Edit Bucket"
                              className="px-3 py-2 rounded-lg text-xs hover:bg-foreground/5 cursor-pointer flex items-center gap-2"
                            >
                              <Edit2 size={13} />
                              <span>Edit Details</span>
                            </Dropdown.Item>
                          )}
                          {!isDefault && (
                            <Dropdown.Item
                              id="delete"
                              textValue="Delete Bucket"
                              variant="danger"
                              className="px-3 py-2 rounded-lg text-xs text-danger hover:bg-danger/10 cursor-pointer flex items-center gap-2"
                            >
                              <Trash2 size={13} />
                              <span>Delete</span>
                            </Dropdown.Item>
                          )}
                        </Dropdown.Menu>
                      </Dropdown.Popover>
                    </Dropdown>
                  </div>
                </div>

                {target > 0 && (
                  <div className="mt-1">
                    <div className="flex justify-between items-center text-xs text-muted-foreground mb-1.5 font-medium">
                      <span>Goal Progress</span>
                      <span className="font-mono">
                        {progress}% of {formatCurrency(target, activeAccountCurrency)}
                      </span>
                    </div>
                    <div className="w-full h-2 bg-foreground/10 rounded-full overflow-hidden">
                      <div
                        className={`h-full ${meta.barColor} transition-all duration-500 rounded-full`}
                        style={{ width: `${progress}%` }}
                      />
                    </div>
                  </div>
                )}
              </div>
            );
          })
        )}
      </div>

      <CreateBucketModal
        isOpen={isCreateOpen}
        onOpenChange={setIsCreateOpen}
      />
      <EditBucketModal
        isOpen={isEditOpen}
        onOpenChange={setIsEditOpen}
        bucket={selectedBucket}
      />
      <DeleteBucketModal
        isOpen={isDeleteOpen}
        onOpenChange={setIsDeleteOpen}
        bucket={selectedBucket}
      />
      {(() => {
        const defaultBucket =
          buckets.find(
            (b: any) =>
              b.type === "DEFAULT" ||
              b.bucket_type === "DEFAULT" ||
              (typeof b.name === "string" &&
                b.name.toLowerCase().includes("unallocated"))
          ) || buckets[0];
        const defaultBucketId =
          defaultBucket?.id ||
          defaultBucket?.bucket_id ||
          defaultBucket?.bucket_uuid;

        return (
          <TransferMoneyModal
            isOpen={isTransferOpen}
            onOpenChange={setIsTransferOpen}
            buckets={buckets}
            defaultSourceId={defaultBucketId}
            defaultTargetId={transferTargetId}
            currency={activeAccountCurrency}
            unassignedBalance={Math.max(0, diff)}
          />
        );
      })()}
    </div>
  );
};
