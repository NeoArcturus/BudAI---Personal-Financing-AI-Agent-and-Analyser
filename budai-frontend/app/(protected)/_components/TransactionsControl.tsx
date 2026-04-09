"use client";

import React, { useState } from "react";
import { Calendar, Search } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";

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
    <div className="flex flex-wrap items-end gap-4 p-4 border-b border-slate-800 bg-[#161B22] z-10 rounded-t-xl">
      <div className="flex-1 min-w-37.5">
        <label className="block text-[10px] font-bold tracking-widest text-slate-500 uppercase mb-2">
          From Date
        </label>
        <div className="relative">
          <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <Input
            type="date"
            value={fromDate}
            onChange={(e) => setFromDate(e.target.value)}
            className="pl-9 bg-[#0D1117] border-slate-700 focus-visible:ring-[#00FFAA]"
          />
        </div>
      </div>

      <div className="flex-1 min-w-37.5">
        <label className="block text-[10px] font-bold tracking-widest text-slate-500 uppercase mb-2">
          To Date
        </label>
        <div className="relative">
          <Calendar className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-slate-500" />
          <Input
            type="date"
            value={toDate}
            onChange={(e) => setToDate(e.target.value)}
            className="pl-9 bg-[#0D1117] border-slate-700 focus-visible:ring-[#00FFAA]"
          />
        </div>
      </div>

      <Button
        onClick={() => onFetchTransactions(fromDate, toDate)}
        disabled={!fromDate || !toDate}
        className="bg-[#00FFAA] text-black hover:bg-[#00FFAA]/80 font-bold tracking-widest uppercase text-xs h-10 px-6"
      >
        <Search className="w-4 h-4 mr-2" /> Fetch Ledger
      </Button>
    </div>
  );
};
