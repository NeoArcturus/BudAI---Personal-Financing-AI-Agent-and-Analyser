// app/dashboard/_components/SidebarLeft.tsx
"use client";

import React from "react";
import {
  CreditCard,
  Globe,
  LogOut,
  Plus,
  AlertTriangle,
  Unlink,
  Loader2,
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
  onOpenLedger: (account: Account) => void; // Kept in interface to prevent page.tsx prop errors, but no longer used on click
  onRevokeAccess: (providerId: string, e: React.MouseEvent) => void;
  onLinkBank: (e: React.MouseEvent) => void;
}

export const SidebarLeft: React.FC<SidebarLeftProps> = ({
  accounts,
  activeAccountId,
  totalBalance,
  userName,
  revokingProviderId,
  onLogout,
  onSetActiveAccount,
  onRevokeAccess,
  onLinkBank,
}) => {
  return (
    <aside className="w-[22%] p-6 flex flex-col gap-4 overflow-y-auto shrink-0 border-r border-slate-800 scrollbar-hide">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xs font-bold tracking-widest text-slate-500 uppercase">
            Welcome Back
          </h2>
          <p className="text-[#00FFAA] font-bold text-lg capitalize">
            {userName}
          </p>
        </div>
        <button
          type="button"
          onClick={onLogout}
          title="Logout"
          className="text-slate-500 hover:text-red-400 transition-colors bg-[#1c2128] p-2 rounded-xl border border-slate-800"
        >
          <LogOut size={16} />
        </button>
      </div>

      <div
        onClick={() => onSetActiveAccount("ALL")}
        className={`bg-[#161B22] p-5 rounded-2xl border ${
          activeAccountId === "ALL"
            ? "border-[#00FFAA]"
            : "border-slate-800 hover:border-[#00FFAA]/50"
        } cursor-pointer transition-all shrink-0 group relative`}
      >
        <div className="flex items-center gap-3 mb-4 pr-8">
          <Globe className="text-[#00FFAA] w-6 h-6 shrink-0 group-hover:scale-110 transition-transform" />
          <span className="text-sm font-bold text-slate-200">All Accounts</span>
        </div>
        <h3 className="text-2xl font-mono font-bold text-white">
          £
          {totalBalance.toLocaleString(undefined, {
            minimumFractionDigits: 2,
            maximumFractionDigits: 2,
          })}
        </h3>
      </div>

      {accounts.map((acc, idx) => {
        const isActive = acc.account_id === activeAccountId;

        return (
          <div
            key={idx}
            // CHANGED: Now this only sets the active account, it no longer pops the ledger modal!
            onClick={() => onSetActiveAccount(acc.account_id)}
            className={`bg-[#161B22] p-5 rounded-2xl border ${
              isActive
                ? "border-[#00FFAA]"
                : "border-slate-800 hover:border-[#00FFAA]/50"
            } cursor-pointer transition-all shrink-0 group relative`}
          >
            <button
              onClick={(e) =>
                onRevokeAccess(acc.provider_id || acc.account_id, e)
              }
              disabled={revokingProviderId === acc.provider_id}
              className="absolute top-4 right-4 text-slate-600 hover:text-red-500 transition-colors bg-[#0D1117] p-1.5 rounded-lg border border-slate-800 disabled:opacity-50"
              title="Disconnect Bank Account"
            >
              {revokingProviderId === acc.provider_id ? (
                <Loader2 size={14} className="animate-spin" />
              ) : (
                <Unlink size={14} />
              )}
            </button>

            <div className="flex items-center gap-3 mb-4 pr-8">
              <CreditCard className="text-[#00FFAA] w-6 h-6 shrink-0 group-hover:scale-110 transition-transform" />
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
        className="w-full mt-2 flex items-center justify-center gap-2 bg-[#161B22] border border-slate-800 text-slate-400 p-4 rounded-2xl hover:border-[#00FFAA]/50 hover:text-[#00FFAA] transition-all border-dashed"
      >
        <Plus size={18} />{" "}
        <span className="text-xs font-bold tracking-widest uppercase">
          Link Bank
        </span>
      </button>
    </aside>
  );
};
