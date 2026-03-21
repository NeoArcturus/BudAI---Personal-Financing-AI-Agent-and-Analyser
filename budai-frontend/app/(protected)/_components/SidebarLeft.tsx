"use client";

import React from "react";
import Link from "next/link";
import { usePathname } from "next/navigation";
import {
  CreditCard,
  LogOut,
  Plus,
  AlertTriangle,
  Unlink,
  Loader2,
  Home,
  PieChart,
  User,
} from "lucide-react";
import { Account } from "@/types";

interface SidebarLeftProps {
  accounts: Account[];
  activeAccountId: string | null;
  totalBalance: number;
  userName: string;
  revokingProviderId: string | null;
  onLogout: () => void;
  onSetActiveAccount: (id: string) => void;
  onOpenLedger: (account: Account) => void;
  onRevokeAccess: (providerId: string, e: React.MouseEvent) => void;
  onLinkBank: (e: React.MouseEvent) => void;
}

export const SidebarLeft: React.FC<SidebarLeftProps> = ({
  accounts,
  activeAccountId,
  userName,
  revokingProviderId,
  onLogout,
  onSetActiveAccount,
  onRevokeAccess,
  onLinkBank,
}) => {
  const pathname = usePathname();
  const isFinancesActive = pathname.includes("/finances");

  return (
    <aside className="w-full h-full p-6 flex flex-col gap-4 overflow-y-auto shrink-0 scrollbar-hide">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xs font-bold tracking-widest text-slate-500 uppercase">
            Welcome Back
          </h2>
          <p className="text-[#69F0AE] font-bold text-lg capitalize">
            {userName}
          </p>
        </div>
        <button
          type="button"
          onClick={onLogout}
          className="text-slate-500 hover:text-red-400 transition-colors bg-[#1A2D21] p-2 rounded-xl border border-[#1A2D21]"
        >
          <LogOut size={16} />
        </button>
      </div>

      <nav className="flex flex-col gap-2 mb-4">
        <Link
          href="/home"
          className={`flex items-center gap-3 p-3 rounded-xl transition-all border ${pathname.includes("/home") ? "bg-[#69F0AE]/10 text-[#69F0AE] border-[#69F0AE]/30" : "text-slate-400 hover:text-white border-transparent hover:bg-[#1A2D21]"}`}
        >
          <Home size={18} />
          <span className="text-sm font-bold">Home</span>
        </Link>
        <Link
          href="/finances"
          className={`flex items-center gap-3 p-3 rounded-xl transition-all border ${pathname.includes("/finances") ? "bg-[#69F0AE]/10 text-[#69F0AE] border-[#69F0AE]/30" : "text-slate-400 hover:text-white border-transparent hover:bg-[#1A2D21]"}`}
        >
          <PieChart size={18} />
          <span className="text-sm font-bold">Finances</span>
        </Link>
        <Link
          href="/profile"
          className={`flex items-center gap-3 p-3 rounded-xl transition-all border ${pathname.includes("/profile") ? "bg-[#69F0AE]/10 text-[#69F0AE] border-[#69F0AE]/30" : "text-slate-400 hover:text-white border-transparent hover:bg-[#1A2D21]"}`}
        >
          <User size={18} />
          <span className="text-sm font-bold">Profile</span>
        </Link>
      </nav>

      {isFinancesActive &&
        accounts.map((acc, idx) => {
          const isActive = acc.account_id === activeAccountId;
          return (
            <div
              key={idx}
              onClick={() => onSetActiveAccount(acc.account_id)}
              className={`bg-[#132017] p-5 rounded-2xl border ${isActive ? "border-[#69F0AE]" : "border-[#1A2D21] hover:border-[#69F0AE]/50"} cursor-pointer transition-all shrink-0 group relative`}
            >
              <button
                onClick={(e) =>
                  onRevokeAccess(acc.provider_id || acc.account_id, e)
                }
                disabled={revokingProviderId === acc.provider_id}
                className="absolute top-4 right-4 text-slate-600 hover:text-red-500 transition-colors bg-[#0A120D] p-1.5 rounded-lg border border-[#1A2D21] disabled:opacity-50"
              >
                {revokingProviderId === acc.provider_id ? (
                  <Loader2 size={14} className="animate-spin" />
                ) : (
                  <Unlink size={14} />
                )}
              </button>
              <div className="flex items-center gap-3 mb-4 pr-8">
                <CreditCard className="text-[#69F0AE] w-6 h-6 shrink-0 group-hover:scale-110 transition-transform" />
                <div className="flex flex-col truncate">
                  <span className="text-sm font-bold text-slate-200 truncate">
                    {acc.provider_name || acc.bank_name}
                  </span>
                  <span className="text-[10px] text-slate-500 font-mono tracking-widest mt-0.5">
                    {acc.sort_code} | ••••{acc.account_number}
                  </span>
                </div>
              </div>
              <h3 className="text-2xl font-mono font-bold text-white">
                {acc.currency === "GBP" ? "£" : acc.currency}
                {(acc.balance ?? acc.account_balance ?? 0).toLocaleString(
                  undefined,
                  { minimumFractionDigits: 2, maximumFractionDigits: 2 },
                )}
              </h3>
              {acc.status === "revoked" && (
                <div className="mt-4 flex items-center justify-center gap-2 py-2 bg-red-500/10 text-red-400 border border-red-500/50 rounded-lg text-xs font-bold">
                  <AlertTriangle size={14} /> Reconnect Required
                </div>
              )}
            </div>
          );
        })}

      <button
        type="button"
        onClick={onLinkBank}
        className="w-full mt-2 flex items-center justify-center gap-2 bg-[#132017] border border-[#1A2D21] text-slate-400 p-4 rounded-2xl hover:border-[#69F0AE]/50 hover:text-[#69F0AE] transition-all border-dashed"
      >
        <Plus size={18} />{" "}
        <span className="text-xs font-bold tracking-widest uppercase">
          Link Bank
        </span>
      </button>
    </aside>
  );
};
