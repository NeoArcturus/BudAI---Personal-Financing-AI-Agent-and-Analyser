"use client";

import React from "react";
import { X, Receipt } from "lucide-react";
import { Transaction } from "@/types";
import TransactionFeed from "./TransactionFeed";

interface TransactionModalProps {
  isOpen: boolean;
  onClose: () => void;
  transactions: Transaction[];
  bankName: string;
}

/*
 * Modal designated exclusively for displaying account transactions.
 * Keeps the main dashboard clean for high-level charts and AI chat.
 */
export default function TransactionModal({
  isOpen,
  onClose,
  transactions,
  bankName,
}: TransactionModalProps) {
  if (!isOpen) return null;

  return (
    <div className="fixed inset-0 bg-black/80 flex items-center justify-center z-50 p-6 backdrop-blur-sm">
      <div className="bg-[#161B22] border border-slate-700 rounded-3xl w-full max-w-4xl h-[85vh] flex flex-col shadow-2xl overflow-hidden animate-in fade-in slide-in-from-bottom-8">
        <div className="p-6 border-b border-slate-800 flex justify-between items-center bg-[#1c2128]">
          <h2 className="text-lg font-bold text-white tracking-widest uppercase flex items-center gap-3">
            <Receipt className="text-[#00FFAA]" size={20} />
            {bankName} Ledger
          </h2>
          <button
            onClick={onClose}
            className="text-slate-400 hover:text-white transition-colors bg-slate-800/50 hover:bg-slate-800 p-2 rounded-full"
          >
            <X size={18} />
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-6 scrollbar-hide bg-[#0D1117]">
          <TransactionFeed transactions={transactions} showCategory={true} />
        </div>
      </div>
    </div>
  );
}
