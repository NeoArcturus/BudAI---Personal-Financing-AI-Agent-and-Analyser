"use client";

import React, { useState } from "react";
import { Modal, Button, Dropdown, toast } from "@heroui/react";
import { ChevronDown, ArrowRight } from "lucide-react";
import { apiFetch } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";
import { formatCurrency, getCurrencySymbol } from "@/lib/currency";

interface TransferMoneyModalProps {
  isOpen: boolean;
  onOpenChange: (open: boolean) => void;
  buckets: any[];
  defaultSourceId?: string;
  defaultTargetId?: string;
  currency?: string;
  unassignedBalance?: number;
}

export const TransferMoneyModal = ({
  isOpen,
  onOpenChange,
  buckets = [],
  defaultSourceId,
  defaultTargetId,
  currency = "GBP",
  unassignedBalance,
}: TransferMoneyModalProps) => {
  const queryClient = useQueryClient();
  const [sourceId, setSourceId] = useState<string>("");
  const [targetId, setTargetId] = useState<string>("");
  const [amount, setAmount] = useState<string>("");
  const [isSubmitting, setIsSubmitting] = useState<boolean>(false);

  // Helper to extract identifier across varying schema keys
  const getBid = React.useCallback(
    (b: any): string => b?.id || b?.bucket_id || b?.bucket_uuid || "",
    []
  );

  // Compute effective balance for any bucket representation (respecting unallocated funds)
  const getBucketBalance = React.useCallback(
    (b: any): number => {
      if (!b) return 0;
      const isDefault =
        b.type === "DEFAULT" ||
        b.bucket_type === "DEFAULT" ||
        (typeof b.name === "string" &&
          b.name.toLowerCase().includes("unallocated"));
      if (isDefault) {
        if (typeof b.cached_balance === "number" && b.cached_balance > 0) {
          return b.cached_balance;
        }
        return typeof unassignedBalance === "number"
          ? Math.max(0, unassignedBalance)
          : (b.cached_balance || 0);
      }
      return b.cached_balance || 0;
    },
    [unassignedBalance]
  );

  React.useEffect(() => {
    if (!isOpen) {
      setAmount("");
      return;
    }

    if (!buckets || buckets.length === 0) return;

    const defaultBkt =
      buckets.find(
        (b) =>
          b.type === "DEFAULT" ||
          b.bucket_type === "DEFAULT" ||
          (typeof b.name === "string" &&
            b.name.toLowerCase().includes("unallocated"))
      ) || buckets[0];
    const defaultBktId = getBid(defaultBkt);

    // 1. Resolve Target
    let nextTargetId = defaultTargetId;
    if (nextTargetId && !buckets.some((b) => getBid(b) === nextTargetId)) {
      nextTargetId = undefined;
    }

    // 2. Resolve Source
    let nextSourceId = defaultSourceId;
    if (nextSourceId && !buckets.some((b) => getBid(b) === nextSourceId)) {
      nextSourceId = undefined;
    }

    // If target was specified as the default bucket, source should be another bucket
    if (nextTargetId && nextTargetId === defaultBktId) {
      nextSourceId =
        nextSourceId ||
        getBid(
          buckets.find((b) => getBid(b) !== nextTargetId) ||
            buckets[1] ||
            buckets[0]
        );
    } else {
      // Standard allocation flow: sweep from default unallocated/daily spending
      nextSourceId = nextSourceId || defaultBktId;
    }

    // If target is still missing or equals source, pick first available alternate
    if (!nextTargetId || nextTargetId === nextSourceId) {
      const alternate =
        buckets.find((b) => getBid(b) !== nextSourceId) ||
        buckets[1] ||
        buckets[0];
      nextTargetId = getBid(alternate);
    }

    setSourceId(nextSourceId || "");
    setTargetId(nextTargetId || "");
  }, [isOpen, defaultSourceId, defaultTargetId, buckets, getBid]);

  const sourceBucket = buckets.find((b) => getBid(b) === sourceId);
  const targetBucket = buckets.find((b) => getBid(b) === targetId);

  const sourceBal = getBucketBalance(sourceBucket);
  const targetBal = getBucketBalance(targetBucket);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    const num = parseFloat(amount);
    if (isNaN(num) || num <= 0) {
      toast("Please enter a valid transfer amount.", { variant: "danger" } as any);
      return;
    }

    if (!sourceId || !targetId || sourceId === targetId) {
      toast("Please choose different source and destination buckets.", {
        variant: "danger",
      } as any);
      return;
    }

    if (sourceBucket && sourceBal < num) {
      toast(
        `Insufficient funds in ${sourceBucket.name || "source bucket"}. Available: ${formatCurrency(sourceBal, currency)}`,
        { variant: "danger" } as any,
      );
      return;
    }

    setIsSubmitting(true);
    try {
      const res = await apiFetch(
        "/api/buckets/transfer",
        {
          method: "POST",
          body: JSON.stringify({
            source_bucket_id: sourceId,
            target_bucket_id: targetId,
            amount: num,
          }),
        },
        true,
      );

      if (res.ok) {
        toast(`Successfully moved ${formatCurrency(num, currency)}.`);
        queryClient.invalidateQueries({ queryKey: ["buckets"] });
        setAmount("");
        onOpenChange(false);
      } else {
        const errJson = (await res.json().catch(() => null)) as { detail?: string } | null;
        toast(
          errJson?.detail || "Transfer was not processed. Please check your balances.",
          { variant: "danger" } as any,
        );
      }
    } catch {
      toast("Network connection failed during transfer.", {
        variant: "danger",
      } as any);
    } finally {
      setIsSubmitting(false);
    }
  };

  const sym = getCurrencySymbol(currency);

  return (
    <Modal.Backdrop isOpen={isOpen} onOpenChange={onOpenChange} variant="blur">
      <Modal.Container placement="center">
        <Modal.Dialog className="modal relative w-full max-w-md flex flex-col overflow-hidden pointer-events-auto bg-popover border border-border shadow-2xl rounded-2xl p-6">
          <div className="flex items-center justify-between border-b border-border/50 pb-4 mb-4">
            <div>
              <h2 className="text-base font-bold text-foreground">Move Money</h2>
              <p className="text-xs text-muted-foreground mt-0.5">
                Reallocate funds between your money buckets
              </p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="flex flex-col gap-4">
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-3 items-center">
              <div>
                <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider block mb-1.5">
                  From
                </label>
                <Dropdown>
                  <Button
                    variant="secondary"
                    className="w-full flex items-center justify-between px-3 py-2 h-auto rounded-xl bg-foreground/5 hover:bg-foreground/10 border border-border text-xs font-semibold transition-all text-left shadow-none"
                  >
                    <div className="flex flex-col truncate">
                      <span className="truncate">
                        {sourceBucket?.name || "Select Bucket"}
                      </span>
                      <span className="text-[10px] text-muted-foreground font-mono">
                        {formatCurrency(sourceBal, currency)}
                      </span>
                    </div>
                    <ChevronDown size={14} className="text-muted-foreground shrink-0" />
                  </Button>
                  <Dropdown.Popover className="bg-popover border border-border shadow-xl rounded-xl w-56 p-1.5 z-50">
                    <Dropdown.Menu onAction={(key) => setSourceId(String(key))}>
                      {buckets.map((b) => {
                        const bid = b.id || b.bucket_id || b.bucket_uuid;
                        const bBal = getBucketBalance(b);
                        return (
                          <Dropdown.Item
                            key={bid}
                            id={bid}
                            textValue={b.name}
                            className="px-3 py-2 rounded-lg text-xs hover:bg-foreground/5 cursor-pointer"
                          >
                            <div className="flex justify-between items-center w-full">
                              <span className="font-medium truncate">{b.name}</span>
                              <span className="text-[10px] text-muted-foreground font-mono ml-2 shrink-0">
                                {formatCurrency(bBal, currency)}
                              </span>
                            </div>
                          </Dropdown.Item>
                        );
                      })}
                    </Dropdown.Menu>
                  </Dropdown.Popover>
                </Dropdown>
              </div>

              <div>
                <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider block mb-1.5">
                  To
                </label>
                <Dropdown>
                  <Button
                    variant="secondary"
                    className="w-full flex items-center justify-between px-3 py-2 h-auto rounded-xl bg-foreground/5 hover:bg-foreground/10 border border-border text-xs font-semibold transition-all text-left shadow-none"
                  >
                    <div className="flex flex-col truncate">
                      <span className="truncate">
                        {targetBucket?.name || "Select Bucket"}
                      </span>
                      <span className="text-[10px] text-muted-foreground font-mono">
                        {formatCurrency(targetBal, currency)}
                      </span>
                    </div>
                    <ChevronDown size={14} className="text-muted-foreground shrink-0" />
                  </Button>
                  <Dropdown.Popover className="bg-popover border border-border shadow-xl rounded-xl w-56 p-1.5 z-50">
                    <Dropdown.Menu onAction={(key) => setTargetId(String(key))}>
                      {buckets.map((b) => {
                        const bid = b.id || b.bucket_id || b.bucket_uuid;
                        const bBal = getBucketBalance(b);
                        return (
                          <Dropdown.Item
                            key={bid}
                            id={bid}
                            textValue={b.name}
                            className="px-3 py-2 rounded-lg text-xs hover:bg-foreground/5 cursor-pointer"
                          >
                            <div className="flex justify-between items-center w-full">
                              <span className="font-medium truncate">{b.name}</span>
                              <span className="text-[10px] text-muted-foreground font-mono ml-2 shrink-0">
                                {formatCurrency(bBal, currency)}
                              </span>
                            </div>
                          </Dropdown.Item>
                        );
                      })}
                    </Dropdown.Menu>
                  </Dropdown.Popover>
                </Dropdown>
              </div>
            </div>

            {buckets.length <= 1 && (
              <div className="p-3 rounded-xl bg-amber-500/10 border border-amber-500/20 text-xs text-amber-500 font-medium">
                Create a category or goal first to allocate your unassigned funds.
              </div>
            )}

            <div>
              <div className="flex items-center justify-between mb-1.5">
                <label className="text-[11px] font-semibold text-muted-foreground uppercase tracking-wider block">
                  Transfer Amount
                </label>
                {sourceBucket && (
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    onPress={() => {
                      const bal = Math.max(0, sourceBal);
                      setAmount(bal > 0 ? bal.toFixed(2) : "");
                    }}
                    className="h-auto p-0 text-[11px] font-mono text-primary hover:underline hover:bg-transparent"
                  >
                    Available: {formatCurrency(sourceBal, currency)}
                  </Button>
                )}
              </div>
              <div className="relative flex items-center bg-foreground/5 border border-border rounded-xl focus-within:border-primary/50 transition-colors">
                <span className="text-foreground/50 font-bold text-sm pl-3 absolute left-0">
                  {sym}
                </span>
                <input
                  type="number"
                  step="0.01"
                  min="0.01"
                  placeholder="0.00"
                  value={amount}
                  onChange={(e) => setAmount(e.target.value)}
                  className="w-full bg-transparent text-foreground font-bold text-sm py-2.5 pl-8 pr-3 outline-none"
                  autoFocus
                />
              </div>
            </div>

            <div className="flex items-center justify-end gap-2 pt-2 border-t border-border/50">
              <Button
                variant="ghost"
                onPress={() => onOpenChange(false)}
                className="text-xs font-semibold px-4 py-2 rounded-xl"
              >
                Cancel
              </Button>
              <Button
                type="submit"
                isDisabled={buckets.length <= 1 || isSubmitting}
                isPending={isSubmitting}
                className="bg-primary text-primary-foreground text-xs font-bold px-5 py-2 rounded-xl shadow-sm hover:opacity-90 transition-opacity"
              >
                Confirm Move
              </Button>
            </div>
          </form>
        </Modal.Dialog>
      </Modal.Container>
    </Modal.Backdrop>
  );
};
