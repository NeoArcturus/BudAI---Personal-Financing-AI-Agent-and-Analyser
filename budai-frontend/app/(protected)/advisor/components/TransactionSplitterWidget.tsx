"use client";

import React, { useState } from "react";
import { Card, Button } from "@heroui/react";
import { Check, Split, Plus, Trash2 } from "lucide-react";

interface Split {
  category: string;
  amount: number;
}

interface TransactionSplitterWidgetProps {
  transactionId: string;
  originalAmount: number;
  merchantName: string;
  suggestedSplits: Split[];
  onConfirm?: (splits: Split[]) => void;
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
  const [splits, setSplits] = useState<Split[]>(
    suggestedSplits.length > 0 
      ? suggestedSplits 
      : [{ category: "General", amount: originalAmount }]
  );

  const [confirmed, setConfirmed] = useState(false);

  const handleAmountChange = (index: number, newAmount: number) => {
    const newSplits = [...splits];
    newSplits[index].amount = newAmount;
    
    // Auto-adjust the last split to maintain the total if possible
    const currentTotal = newSplits.reduce((acc, s, i) => i !== newSplits.length - 1 ? acc + s.amount : acc, 0);
    const remainder = originalAmount - currentTotal;
    
    if (index !== newSplits.length - 1 && remainder >= 0) {
      newSplits[newSplits.length - 1].amount = parseFloat(remainder.toFixed(2));
    }
    
    setSplits(newSplits);
  };

  const handleConfirm = () => {
    setConfirmed(true);
    if (onConfirm) onConfirm(splits);
  };

  const currentTotal = splits.reduce((acc, s) => acc + s.amount, 0);
  const isTotalValid = Math.abs(currentTotal - originalAmount) < 0.01;

  if (confirmed) {
    return (
      <Card className="w-full max-w-sm border-success/30 my-2">
        <div className="p-4 flex flex-row items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-success/20 flex items-center justify-center text-success">
            <Check size={16} />
          </div>
          <div>
            <p className="text-sm font-bold text-foreground">Split Confirmed</p>
            <p className="text-[10px] text-foreground/50 uppercase tracking-widest">{merchantName}</p>
          </div>
        </div>
      </Card>
    );
  }

  return (
    <Card className="w-full max-w-md overflow-hidden my-4">
      <div className="flex flex-col items-start px-5 pt-5 pb-0">
        <div className="flex items-center gap-2 text-primary mb-1">
          <Split size={14} />
          <h3 className="text-xs font-black uppercase tracking-widest">Split Transaction</h3>
        </div>
        <p className="text-sm font-bold mt-2">{merchantName}</p>
        <p className="text-xs text-foreground/50 mt-1">Total: £{originalAmount.toFixed(2)}</p>
      </div>
      
      <div className="px-5 py-4 flex flex-col gap-4">
        {splits.map((split, idx) => (
          <div key={idx} className="flex flex-col gap-2 p-3 bg-black/20 rounded-xl border border-white/5">
            <div className="flex items-center justify-between">
              <div className="w-1/2 flex flex-col gap-1">
                <label className="text-xs">Category</label>
                <input 
                  className="bg-transparent border border-white/10 rounded px-2 py-1 text-[11px] font-bold"
                  value={split.category}
                  onChange={(e) => {
                    const newSplits = [...splits];
                    newSplits[idx].category = e.target.value;
                    setSplits(newSplits);
                  }}
                />
              </div>
              <div className="w-1/3 flex flex-col gap-1">
                <label className="text-xs">Amount (£)</label>
                <input 
                  type="number"
                  className="bg-transparent border border-white/10 rounded px-2 py-1 text-[11px] font-mono text-right"
                  value={split.amount.toString()}
                  onChange={(e) => handleAmountChange(idx, parseFloat(e.target.value) || 0)}
                />
              </div>
              {splits.length > 1 && (
                <Button 
                  isIconOnly 
                   
                  variant="ghost" 
                  onPress={() => setSplits(splits.filter((_, i) => i !== idx))}
                >
                  <Trash2 size={14} />
                </Button>
              )}
            </div>
          </div>
        ))}

        <Button 
           
          variant="ghost"
          onPress={() => setSplits([...splits, { category: "New", amount: 0 }])}
          className="border border-white/10 bg-white/5 text-[10px] uppercase tracking-widest font-bold flex gap-2 items-center"
        >
          <Plus size={14} /> Add Split
        </Button>
      </div>
      
      <div className="px-5 pb-5 pt-0 flex flex-col gap-2">
        <div className="flex justify-between items-center w-full px-1 text-xs mb-2">
          <span className="text-foreground/50">Allocated:</span>
          <span className={`font-mono font-bold ${isTotalValid ? "text-success" : "text-danger"}`}>
            £{currentTotal.toFixed(2)} / £{originalAmount.toFixed(2)}
          </span>
        </div>
        
        <div className="flex w-full gap-2">
          <Button 
            variant="ghost"
            onPress={onCancel}
            className="flex-1 bg-white/5 text-[11px] uppercase tracking-widest font-bold"
          >
            Cancel
          </Button>
          <Button 
            onPress={handleConfirm}
            isDisabled={!isTotalValid}
            className="flex-1 text-[11px] uppercase tracking-widest font-black shadow-[0_0_15px_rgba(0,242,255,0.4)]"
          >
            Confirm Split
          </Button>
        </div>
      </div>
    </Card>
  );
};
