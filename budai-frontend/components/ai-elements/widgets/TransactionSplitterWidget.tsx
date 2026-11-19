"use client";

import React, { useState } from "react";
import { Check, Split, Plus, Trash2 } from "lucide-react";
import { cn } from "@/lib/utils";

interface SplitItem {
  category: string;
  amount: number;
}

export interface TransactionSplitterWidgetProps {
  transactionId: string;
  originalAmount: number;
  merchantName: string;
  suggestedSplits: SplitItem[];
  onConfirm?: (splits: SplitItem[]) => void;
  onCancel?: () => void;
}

export const TransactionSplitterWidget = ({
  transactionId,
  originalAmount,
  merchantName,
  suggestedSplits,
  onConfirm,
  onCancel,
}: TransactionSplitterWidgetProps) => {
  const [splits, setSplits] = useState<SplitItem[]>(
    suggestedSplits.length > 0
      ? suggestedSplits
      : [{ category: "General", amount: originalAmount }]
  );

  const [confirmed, setConfirmed] = useState(false);

  const handleAmountChange = (index: number, newAmount: number) => {
    const newSplits = [...splits];
    newSplits[index].amount = newAmount;

    const currentTotal = newSplits.reduce(
      (acc, s, i) => (i !== newSplits.length - 1 ? acc + s.amount : acc),
      0
    );
    const remainder = originalAmount - currentTotal;

    if (index !== newSplits.length - 1 && remainder >= 0) {
      newSplits[newSplits.length - 1].amount = parseFloat(remainder.toFixed(2));
    }

    setSplits(newSplits);
  };

  const handleCategoryChange = (index: number, newCategory: string) => {
    const newSplits = [...splits];
    newSplits[index].category = newCategory;
    setSplits(newSplits);
  };

  const addSplit = () => {
    const currentTotal = splits.reduce((acc, s) => acc + s.amount, 0);
    const remainder = Math.max(0, originalAmount - currentTotal);
    setSplits([...splits, { category: "General", amount: parseFloat(remainder.toFixed(2)) }]);
  };

  const removeSplit = (index: number) => {
    if (splits.length <= 1) return;
    const newSplits = splits.filter((_, i) => i !== index);
    const currentTotal = newSplits.reduce(
      (acc, s, i) => (i !== newSplits.length - 1 ? acc + s.amount : acc),
      0
    );
    newSplits[newSplits.length - 1].amount = parseFloat(
      (originalAmount - currentTotal).toFixed(2)
    );
    setSplits(newSplits);
  };

  const currentTotal = splits.reduce((acc, s) => acc + s.amount, 0);
  const isValid = Math.abs(currentTotal - originalAmount) < 0.01;

  if (confirmed) {
    return (
      <div className="w-full p-4 rounded-2xl bg-success/10 border border-success/20 text-success flex items-center gap-3">
        <Check size={18} />
        <span className="text-xs font-semibold">
          Transaction allocation verified and applied across {splits.length} categories.
        </span>
      </div>
    );
  }

  return (
    <div className="w-full p-4 rounded-2xl bg-[#0d0e14]/60 backdrop-blur-xl border border-white/10 shadow-xl flex flex-col gap-3">
      <div className="flex items-center justify-between pb-2 border-b border-white/5">
        <div className="flex items-center gap-2">
          <Split size={15} className="text-primary" />
          <span className="text-xs font-bold text-foreground">
            Split Transaction: {merchantName}
          </span>
        </div>
        <span className="text-xs font-mono font-bold text-primary">
          £{originalAmount.toFixed(2)}
        </span>
      </div>

      <div className="space-y-2">
        {splits.map((s, index) => (
          <div key={index} className="flex items-center gap-2 text-xs">
            <input
              type="text"
              value={s.category}
              onChange={(e) => handleCategoryChange(index, e.target.value)}
              placeholder="Category"
              className="flex-1 bg-white/[0.04] border border-white/10 rounded-lg px-2.5 py-1.5 text-foreground outline-none focus:border-primary/40"
            />
            <div className="flex items-center bg-white/[0.04] border border-white/10 rounded-lg px-2.5 py-1.5 w-28">
              <span className="text-muted-foreground mr-1">£</span>
              <input
                type="number"
                step="0.01"
                value={s.amount || ""}
                onChange={(e) =>
                  handleAmountChange(index, parseFloat(e.target.value) || 0)
                }
                className="w-full bg-transparent text-foreground outline-none font-mono"
              />
            </div>
            {splits.length > 1 && (
              <button
                type="button"
                onClick={() => removeSplit(index)}
                className="p-1.5 text-muted-foreground hover:text-danger rounded-md transition-colors"
                aria-label="Remove split"
              >
                <Trash2 size={13} />
              </button>
            )}
          </div>
        ))}
      </div>

      <div className="flex items-center justify-between pt-2 border-t border-white/5 text-xs">
        <button
          type="button"
          onClick={addSplit}
          className="flex items-center gap-1.5 text-primary hover:text-primary/80 transition-colors font-medium"
        >
          <Plus size={13} />
          <span>Add Allocation</span>
        </button>

        <div className="flex items-center gap-2">
          {onCancel && (
            <button
              type="button"
              onClick={onCancel}
              className="px-3 py-1.5 rounded-lg text-muted-foreground hover:text-foreground hover:bg-white/5 transition-all"
            >
              Cancel
            </button>
          )}
          <button
            type="button"
            disabled={!isValid}
            onClick={() => {
              if (isValid) {
                setConfirmed(true);
                onConfirm?.(splits);
              }
            }}
            className={cn(
              "px-3.5 py-1.5 rounded-lg font-semibold transition-all shadow-xs",
              isValid
                ? "bg-primary text-primary-foreground hover:bg-primary/90 cursor-pointer"
                : "bg-white/5 text-muted-foreground cursor-not-allowed"
            )}
          >
            Confirm Allocation
          </button>
        </div>
      </div>
    </div>
  );
};
