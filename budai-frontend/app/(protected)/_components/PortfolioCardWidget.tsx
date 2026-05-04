import React, { useState } from "react";
import Image from "next/image";
import { Account } from "@/types";
import { cn } from "@/lib/utils";

interface PortfolioCardWidgetProps {
  accounts: Account[];
  activeAccountId: string | null;
  onAccountSelect: (id: string) => void;
}

export default function PortfolioCardWidget({
  accounts,
  activeAccountId,
  onAccountSelect,
}: PortfolioCardWidgetProps) {
  const [isExpanded, setIsExpanded] = useState(false);

  const sortedAccounts = [...accounts].sort((a, b) => {
    if (a.account_id === activeAccountId) return -1;
    if (b.account_id === activeAccountId) return 1;
    return 0;
  });

  const displayAccounts = sortedAccounts.slice(0, 4);

  const getGradient = (index: number) => {
    const gradients = [
      "bg-linear-to-br from-indigo-500 via-purple-500 to-pink-500",
      "bg-linear-to-br from-rose-400 via-fuchsia-500 to-indigo-500",
      "bg-linear-to-br from-blue-400 via-emerald-400 to-teal-500",
      "bg-linear-to-br from-orange-400 via-red-500 to-pink-500",
    ];
    return gradients[index % gradients.length];
  };

  const formatSortCode = (sc: string | undefined) => {
    if (!sc) return "00-00-00";
    const cleaned = sc.replace(/\D/g, "");
    if (cleaned.length === 6) {
      return `${cleaned.slice(0, 2)}-${cleaned.slice(2, 4)}-${cleaned.slice(4, 6)}`;
    }
    return sc;
  };

  return (
    <div className="w-full h-full bg-[#13151D]/40 bg-linear-to-br from-white/8 to-transparent backdrop-blur-xl rounded-3xl border border-white/8 p-6 flex flex-col shadow-2xl relative z-10">
      <div className="flex justify-between items-center mb-6">
        <h3 className="text-white font-semibold text-lg tracking-tight">
          Active Account
        </h3>
        <button
          onClick={() => setIsExpanded(!isExpanded)}
          className="text-[#8B8E98] hover:text-white text-sm font-medium transition-colors z-50 relative"
        >
          {isExpanded ? "Collapse \u2191" : "Show All \u2192"}
        </button>
      </div>

      <div className="relative flex-1 w-full min-h-55">
        {displayAccounts.map((acc, idx) => {
          const sortCode = formatSortCode(acc.sort_code);
          const accountNumber = acc.account_number || "****";
          const bankName = acc.bank_name || "Bank";
          const logoUrl = (acc as typeof acc & { logo_url?: string }).logo_url;
          const balance = acc.account_balance ?? acc.balance ?? 0;

          return (
            <div
              key={acc.account_id}
              onClick={() => {
                if (isExpanded) {
                  onAccountSelect(acc.account_id);
                  setIsExpanded(false);
                }
              }}
              className={cn(
                "absolute w-full h-50 rounded-2xl p-6 shadow-xl overflow-hidden transition-all duration-500 ease-out flex flex-col justify-center",
                getGradient(idx),
                isExpanded
                  ? "cursor-pointer hover:ring-2 hover:ring-white/50"
                  : "",
              )}
              style={{
                top: isExpanded ? `${idx * 210}px` : `${idx * 16}px`,
                transform: isExpanded ? `scale(1)` : `scale(${1 - idx * 0.05})`,
                zIndex: 50 - idx,
                opacity: isExpanded ? 1 : 1 - idx * 0.2,
              }}
            >
              <div className="absolute inset-0 bg-black/10 mix-blend-multiply"></div>
              <div className="absolute -right-10 -top-10 w-48 h-48 bg-white/10 rounded-full blur-3xl transition-colors"></div>

              <div className="absolute right-6 top-6 opacity-90 flex items-center gap-3 z-10">
                {logoUrl && (
                  <Image
                    src={logoUrl}
                    alt={bankName}
                    width={40}
                    height={40}
                    className="w-10 h-10 rounded-full bg-white object-contain p-1.5 shadow-md"
                  />
                )}
                <span className="text-white font-bold text-lg">{bankName}</span>
              </div>

              <div className="absolute right-4 top-1/2 -translate-y-1/2 flex gap-1 opacity-40 z-10">
                <div className="w-10 h-10 rounded-full border-2 border-white"></div>
                <div className="w-10 h-10 rounded-full border-2 border-white -ml-5"></div>
              </div>

              <span className="text-white/90 text-sm font-medium mb-1 z-10">
                Account Balance
              </span>
              <h2 className="text-3xl font-bold text-white tracking-tight z-10">
                £{" "}
                {balance.toLocaleString(undefined, {
                  minimumFractionDigits: 2,
                  maximumFractionDigits: 2,
                })}
              </h2>

              <div className="mt-8 flex justify-between items-center text-white/80 text-sm font-medium tracking-widest z-10">
                <span className="flex items-center gap-2">
                  <span>•••• {accountNumber.slice(-4)}</span>
                  <span className="opacity-50">|</span>
                  <span>SC: {sortCode}</span>
                </span>
              </div>
            </div>
          );
        })}
      </div>
    </div>
  );
}
