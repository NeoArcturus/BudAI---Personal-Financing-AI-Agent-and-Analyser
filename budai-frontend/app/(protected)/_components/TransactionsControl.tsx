"use client";

import React, { useState } from "react";
import { Calendar, Search } from "lucide-react";

interface TransactionsControlProps {
  activeAccountId: string | null;
  onFetchTransactions: (fromDate: string, toDate: string) => void;
}

export const TransactionsControl: React.FC<TransactionsControlProps> = ({
  activeAccountId,
  onFetchTransactions,
}) => {
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");

  if (activeAccountId === "ALL" || !activeAccountId) return null;

  return (
    <div className="flex flex-wrap items-end gap-4 p-4 border-b border-slate-800 bg-[#161B22] z-10">
      <div className="flex-1 min-w-37.5">
        <label className="block text-[10px] font-bold tracking-widest text-slate-500 uppercase mb-1">
          From Date
        </label>
        <div className="relative">
          <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="date"
            value={fromDate}
            onChange={(e) => setFromDate(e.target.value)}
            className="w-full bg-[#0D1117] border border-slate-700 rounded-lg py-2 pl-9 pr-3 text-sm text-white focus:outline-none focus:border-[#00FFAA] transition-colors"
          />
        </div>
      </div>

      <div className="flex-1 min-w-37.5">
        <label className="block text-[10px] font-bold tracking-widest text-slate-500 uppercase mb-1">
          To Date
        </label>
        <div className="relative">
          <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <input
            type="date"
            value={toDate}
            onChange={(e) => setToDate(e.target.value)}
            className="w-full bg-[#0D1117] border border-slate-700 rounded-lg py-2 pl-9 pr-3 text-sm text-white focus:outline-none focus:border-[#00FFAA] transition-colors"
          />
        </div>
      </div>

      <button
        onClick={() => onFetchTransactions(fromDate, toDate)}
        disabled={!fromDate || !toDate}
        className="bg-[#1c2128] hover:bg-[#00FFAA]/10 border border-slate-700 hover:border-[#00FFAA] text-[#00FFAA] disabled:bg-[#0D1117] disabled:border-slate-800 disabled:text-slate-600 disabled:hover:bg-[#0D1117] disabled:hover:border-slate-800 py-2 px-6 rounded-lg text-sm font-bold transition-all flex items-center gap-2 h-9.5"
      >
        <Search className="w-4 h-4" />
        Fetch Ledger
      </button>
    </div>
  );
};
