"use client";

import React from "react";
import { useBudAI } from "@/app/context/AppContext";
import { User, Mail, ShieldCheck, Landmark, Building2 } from "lucide-react";

export default function ProfilePage() {
  const { accounts, userName } = useBudAI();

  return (
    <div className="flex flex-col h-full w-full p-8 overflow-y-auto scrollbar-hide gap-8">
      <div>
        <h1 className="text-3xl font-bold text-white mb-2">
          Profile & Security
        </h1>
        <p className="text-slate-400">
          Manage your personal details and connected institutions.
        </p>
      </div>
      <div className="bg-[#132017] border border-[#1A2D21] rounded-2xl p-8 flex items-start gap-8">
        <div className="w-24 h-24 rounded-full bg-[#1A2D21] border-2 border-[#69F0AE] flex items-center justify-center shrink-0">
          <User size={40} className="text-[#69F0AE]" />
        </div>
        <div className="flex flex-col gap-3 flex-1">
          <h2 className="text-2xl font-bold text-white">{userName}</h2>
          <div className="flex items-center gap-2 text-slate-400">
            <Mail size={16} /> user@example.com
          </div>
          <div className="flex items-center gap-2 text-[#69F0AE] text-sm font-bold bg-[#69F0AE]/10 w-fit px-3 py-1 rounded-lg mt-2">
            <ShieldCheck size={16} /> Fully Verified
          </div>
        </div>
      </div>
      <div>
        <h2 className="text-xl font-bold text-white mb-4 flex items-center gap-2">
          <Landmark size={20} className="text-[#69F0AE]" /> Connected
          Institutions
        </h2>
        {accounts.length === 0 ? (
          <div className="bg-[#132017] border border-[#1A2D21] border-dashed rounded-2xl p-12 text-center text-slate-500">
            No accounts connected yet.
          </div>
        ) : (
          <div className="bg-[#132017] border border-[#1A2D21] rounded-2xl overflow-hidden">
            <table className="w-full text-left border-collapse">
              <thead>
                <tr className="bg-[#0A120D] text-slate-500 text-xs uppercase tracking-widest border-b border-[#1A2D21]">
                  <th className="p-4 font-bold text-left">Institution</th>
                  <th className="p-4 font-bold text-center">Account No.</th>
                  <th className="p-4 font-bold text-center">Sort Code</th>
                  <th className="p-4 font-bold text-center">Currency</th>
                  <th className="p-4 font-bold text-right">Balance</th>
                </tr>
              </thead>
              <tbody className="divide-y divide-[#1A2D21]">
                {accounts.map((acc, i) => (
                  <tr
                    key={i}
                    className="hover:bg-[#1A2D21]/30 transition-colors"
                  >
                    <td className="p-4 flex items-center gap-3 text-white font-bold text-left">
                      <div className="p-2 bg-[#1A2D21] rounded-lg text-[#69F0AE]">
                        <Building2 size={16} />
                      </div>
                      {acc.provider_name || acc.bank_name}
                    </td>
                    <td className="p-4 text-slate-300 font-mono text-sm text-center">
                      ••••{acc.account_number}
                    </td>
                    <td className="p-4 text-slate-300 font-mono text-sm text-center">
                      {acc.sort_code}
                    </td>
                    <td className="p-4 text-slate-300 text-sm text-center">
                      {acc.currency}
                    </td>
                    <td className="p-4 text-white font-mono font-bold text-right">
                      {acc.currency === "GBP" ? "£" : acc.currency}
                      {(acc.balance ?? acc.account_balance ?? 0).toLocaleString(
                        undefined,
                        { minimumFractionDigits: 2, maximumFractionDigits: 2 },
                      )}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        )}
      </div>
    </div>
  );
}
