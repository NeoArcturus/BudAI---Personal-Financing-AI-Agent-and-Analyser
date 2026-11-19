"use client";

import React, { useMemo } from "react";
import { Tooltip, Chip } from "@heroui/react";
import { Receipt, Calendar, Building2, Tag, CheckCircle2 } from "lucide-react";
import { formatCurrency } from "@/lib/currency";
import { useQueryClient } from "@tanstack/react-query";
import { Transaction } from "@/types";
import { cn } from "@/lib/utils";

export interface InlineCitationProps {
  txId: string;
  fallbackText?: string;
  className?: string;
}

export function InlineCitation({ txId, fallbackText, className }: InlineCitationProps) {
  const queryClient = useQueryClient();

  // Look for transaction across all cached transaction queries in React Query
  const cachedTx = useMemo(() => {
    if (!txId) return null;
    const queries = queryClient.getQueryCache().findAll({ queryKey: ["transactions"] });
    for (const query of queries) {
      const data = query.state.data as Transaction[] | undefined;
      if (Array.isArray(data)) {
        const found = data.find(
          (t) =>
            t.transaction_id === txId ||
            t.transaction_uuid === txId ||
            (t as any).id === txId
        );
        if (found) return found;
      }
    }
    return null;
  }, [queryClient, txId]);

  // Generate compact reference pill label (last 4 characters of UUID, e.g. [#8253])
  const compactLabel = useMemo(() => {
    if (fallbackText) return fallbackText;
    const clean = txId.replace(/^txn[-_]/i, "");
    const last4 = clean.slice(-4).toUpperCase();
    return `[#${last4}]`;
  }, [txId, fallbackText]);

  const merchant = cachedTx?.merchant_name || cachedTx?.description || "Verified Payment";
  const amount = cachedTx?.amount ?? (cachedTx as any)?.Amount ?? 0;
  const currency = cachedTx?.currency || "GBP";
  const isCredit = amount > 0;
  const dateStr = cachedTx?.date || cachedTx?.timestamp || "";
  const formattedDate = dateStr
    ? new Date(dateStr).toLocaleDateString("en-GB", {
        day: "numeric",
        month: "short",
        year: "numeric",
      })
    : null;
  const category = cachedTx?.category || (cachedTx as any)?.Category || "Verified Record";
  const status = cachedTx?.status || "settled";

  return (
    <Tooltip delay={50} closeDelay={150}>
      <Tooltip.Trigger className="inline-flex items-baseline align-baseline outline-none mx-0.5">
        <span
          className={cn(
            "inline-flex items-center px-1.5 py-0.2 rounded-md font-mono text-[10px] font-semibold text-primary/90 bg-primary/10 border border-primary/20 hover:bg-primary/20 hover:border-primary/40 transition-colors cursor-pointer select-none",
            className
          )}
        >
          {compactLabel}
        </span>
      </Tooltip.Trigger>
      <Tooltip.Content
        showArrow
        offset={8}
        className="w-72 bg-[#0b0c14]/90 backdrop-blur-2xl border border-white/20 p-3 rounded-2xl shadow-2xl text-xs z-50 text-foreground"
      >
        <Tooltip.Arrow className="fill-[#0b0c14]/90 stroke-white/20" />
        <div className="flex flex-col gap-2">
          <div className="flex items-start justify-between gap-2">
            <div className="flex flex-col gap-0.5 min-w-0">
              <span className="font-bold text-xs text-foreground truncate flex items-center gap-1.5">
                <Receipt size={12} className="text-primary shrink-0" />
                <span className="truncate">{merchant}</span>
              </span>
              <span className="text-[10px] text-muted-foreground font-mono truncate">
                Ref: {txId.slice(0, 16)}...
              </span>
            </div>
            {cachedTx && (
              <span
                className={`font-mono font-bold text-xs shrink-0 px-2 py-0.5 rounded-md ${
                  isCredit
                    ? "bg-emerald-500/15 text-emerald-400 border border-emerald-500/30"
                    : "bg-white/10 text-foreground border border-white/15"
                }`}
              >
                {isCredit ? "+" : ""}
                {formatCurrency(amount, currency)}
              </span>
            )}
          </div>

          {cachedTx && (
            <div className="grid grid-cols-2 gap-1 pt-1.5 border-t border-white/10 text-[10px]">
              {formattedDate && (
                <div className="flex items-center gap-1 text-muted-foreground">
                  <Calendar size={10} />
                  <span>{formattedDate}</span>
                </div>
              )}
              <div className="flex items-center gap-1 text-muted-foreground">
                <Tag size={10} />
                <span className="truncate">{category}</span>
              </div>
              <div className="flex items-center gap-1 text-emerald-400 col-span-2">
                <CheckCircle2 size={10} />
                <span className="capitalize">{status}</span>
              </div>
            </div>
          )}
        </div>
      </Tooltip.Content>
    </Tooltip>
  );
}
