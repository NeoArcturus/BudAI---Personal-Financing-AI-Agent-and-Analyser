import React, { useState } from "react";
import { Calendar, Filter } from "lucide-react";
import { Transaction } from "@/types";
import { Button, Input, Chip, ScrollShadow } from "@heroui/react";
import { cn } from "@/lib/utils";

interface LedgerTableWidgetProps {
  transactions: Transaction[];
  activeAccountId: string | null;
  onFilter: (from: string, to: string) => void;
}

export default function LedgerTableWidget({
  transactions,
  activeAccountId,
  onFilter,
}: LedgerTableWidgetProps) {
  const [fromDate, setFromDate] = useState("");
  const [toDate, setToDate] = useState("");

  const getCategoryTheme = (category: string) => {
    const themes: Record<string, string> = {
      Income: "bg-[#00E096]/10 text-[#00E096] border-[#00E096]/20",
      Subscriptions: "bg-[#3D73FF]/10 text-[#3D73FF] border-[#3D73FF]/20",
      "Food and dining": "bg-[#FF5E98]/10 text-[#FF5E98] border-[#FF5E98]/20",
      Bills: "bg-[#FF8A4C]/10 text-[#FF8A4C] border-[#FF8A4C]/20",
    };
    return (
      themes[category] || "bg-[#8B8E98]/10 text-[#8B8E98] border-[#8B8E98]/20"
    );
  };

  const getInitialColor = (initial: string) => {
    const colors = [
      "bg-[#00E096]",
      "bg-[#3D73FF]",
      "bg-[#FF5E98]",
      "bg-[#FF8A4C]",
      "bg-[#9333EA]",
    ];
    const charCode = initial.charCodeAt(0) || 0;
    return colors[charCode % colors.length];
  };

  const formatShortDate = (dateStr: string) => {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    const day = d.toLocaleDateString("en-US", { weekday: "short" });
    const time = d
      .toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })
      .toLowerCase();
    return `${day} ${time}`;
  };

  if (activeAccountId === "ALL" || !activeAccountId) return null;

  return (
    <div className="w-full h-full bg-[#13151D]/40 bg-linear-to-br from-white/8 to-transparent backdrop-blur-xl rounded-3xl border border-white/8 flex flex-col shadow-2xl overflow-hidden">
      <div className="flex flex-wrap items-center justify-between p-6 border-b border-white/5 shrink-0 gap-4">
        <h3 className="text-white font-semibold text-lg tracking-tight">
          Transaction history
        </h3>
        <div className="flex items-center gap-3">
          <div className="flex items-center bg-[#181A20] border border-white/5 rounded-xl p-1 shadow-inner">
            <Calendar className="w-4 h-4 text-[#5E6272] ml-3" />
            <Input
              aria-label="From Date"
              type="date"
              value={fromDate}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setFromDate(e.target.value)
              }
              className="w-32 [&>div]:bg-transparent [&>div]:data-[hover=true]:bg-transparent [&>div]:group-data-[focus=true]:bg-transparent [&>div]:shadow-none [&>div]:border-none [&>div]:p-0 [&>div]:min-h-0 [&>div]:h-auto [&_input]:text-sm [&_input]:text-[#8B8E98] [&_input::-webkit-calendar-picker-indicator]:invert-[0.6]"
            />
            <span className="text-[#5E6272]">|</span>
            <Input
              aria-label="To Date"
              type="date"
              value={toDate}
              onChange={(e: React.ChangeEvent<HTMLInputElement>) =>
                setToDate(e.target.value)
              }
              className="w-32 [&>div]:bg-transparent [&>div]:data-[hover=true]:bg-transparent [&>div]:group-data-[focus=true]:bg-transparent [&>div]:shadow-none [&>div]:border-none [&>div]:p-0 [&>div]:min-h-0 [&>div]:h-auto [&_input]:text-sm [&_input]:text-[#8B8E98] [&_input::-webkit-calendar-picker-indicator]:invert-[0.6]"
            />
          </div>
          <Button
            onPress={() => onFilter(fromDate, toDate)}
            isDisabled={!fromDate || !toDate}
            variant="primary"
            className="bg-white/5 hover:bg-white/10 border-white/5 text-white h-10 px-5 rounded-xl border data-[disabled=true]:opacity-50 min-w-0"
          >
            <Filter className="w-4 h-4 mr-2" /> Apply filter
          </Button>
        </div>
      </div>

      <ScrollShadow className="flex-1 h-full w-full">
        <div className="px-2 pb-4">
          {transactions.length === 0 ? (
            <div className="flex items-center justify-center h-48 text-[#5E6272] text-sm font-medium">
              No transactions found for this period.
            </div>
          ) : (
            <table className="w-full text-left border-collapse relative">
              <thead className="sticky top-0 bg-[#13151D]/90 backdrop-blur-md z-10">
                <tr className="text-[#5E6272] text-xs font-medium border-b border-white/5">
                  <th className="py-4 px-6 font-medium">Transaction</th>
                  <th className="py-4 px-6 font-medium">Amount</th>
                  <th className="py-4 px-6 font-medium">Date</th>
                  <th className="py-4 px-6 font-medium">Category</th>
                </tr>
              </thead>
              <tbody>
                {transactions.map((tx, i) => {
                  const desc = tx.description || tx.Description || "Unknown";
                  const amount = tx.amount ?? tx.Amount ?? 0;
                  const cat = tx.category || tx.Category || "Uncategorized";
                  const initial = desc.charAt(0).toUpperCase();
                  const isPositive = amount > 0 || cat === "Income";

                  return (
                    <tr
                      key={i}
                      className="border-b border-white/5 last:border-0 hover:bg-white/5 transition-colors group"
                    >
                      <td className="py-4 px-6 max-w-75">
                        <div className="flex items-center gap-4">
                          <div
                            className={cn(
                              "w-9 h-9 rounded-full flex items-center justify-center text-white font-bold text-sm shrink-0 shadow-lg",
                              getInitialColor(initial),
                            )}
                          >
                            {initial}
                          </div>
                          <span className="text-white text-sm font-medium truncate block w-full group-hover:text-white/90 transition-colors">
                            {desc}
                          </span>
                        </div>
                      </td>
                      <td className="py-4 px-6">
                        <span
                          className={cn(
                            "text-sm font-medium",
                            isPositive ? "text-[#00E096]" : "text-[#8B8E98]",
                          )}
                        >
                          {isPositive ? "+" : "-"} £{" "}
                          {Math.abs(amount).toFixed(2)}
                        </span>
                      </td>
                      <td className="py-4 px-6 text-sm text-[#8B8E98] whitespace-nowrap">
                        {formatShortDate(tx.timestamp || tx.date || "")}
                      </td>
                      <td className="py-4 px-6">
                        <Chip
                          className={cn(
                            "px-2 text-xs font-medium border",
                            getCategoryTheme(cat),
                          )}
                        >
                          {cat}
                        </Chip>
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          )}
        </div>
      </ScrollShadow>
    </div>
  );
}
