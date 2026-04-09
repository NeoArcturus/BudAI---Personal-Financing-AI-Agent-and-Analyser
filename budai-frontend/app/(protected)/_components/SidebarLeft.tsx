"use client";

import React, { useState, useEffect } from "react";
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
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";

// ... (keep the SidebarLeftProps interface exactly the same) ...
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
  const [isMounted, setIsMounted] = useState(false);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsMounted(true);
    }, 0);

    return () => clearTimeout(timer);
  }, []);

  return (
    <aside className="w-full h-full p-6 flex flex-col gap-4 overflow-y-auto shrink-0 scrollbar-hide bg-[#0D1117] border-r border-slate-800">
      <div className="flex justify-between items-center mb-6">
        <div>
          <h2 className="text-xs font-bold tracking-widest text-slate-500 uppercase">
            Welcome Back
          </h2>
          <p className="text-[#00FFAA] font-bold text-lg capitalize">
            {isMounted ? userName : "User"}
          </p>
        </div>
        <Button
          variant="ghost"
          size="icon"
          onClick={onLogout}
          className="text-slate-500 hover:text-red-400 hover:bg-red-400/10"
        >
          <LogOut size={16} />
        </Button>
      </div>

      <nav className="flex flex-col gap-2 mb-4">
        {[
          { href: "/home", icon: Home, label: "Home" },
          { href: "/finances", icon: PieChart, label: "Finances" },
          { href: "/profile", icon: User, label: "Profile" },
        ].map((item) => (
          <Button
            key={item.href}
            variant={pathname.includes(item.href) ? "secondary" : "ghost"}
            className={`w-full justify-start gap-3 h-12 ${
              pathname.includes(item.href)
                ? "bg-[#00FFAA]/10 text-[#00FFAA] hover:bg-[#00FFAA]/20"
                : "text-slate-400 hover:text-white"
            }`}
            asChild
          >
            <Link href={item.href}>
              <item.icon size={18} />
              <span className="font-bold">{item.label}</span>
            </Link>
          </Button>
        ))}
      </nav>

      {isFinancesActive &&
        accounts.map((acc, idx) => {
          const isActive = acc.account_id === activeAccountId;
          return (
            <Card
              key={idx}
              onClick={() => onSetActiveAccount(acc.account_id)}
              className={`bg-[#161B22] border-slate-800 cursor-pointer transition-all shrink-0 group relative overflow-hidden ${
                isActive
                  ? "ring-1 ring-[#00FFAA] border-transparent"
                  : "hover:border-[#00FFAA]/50"
              }`}
            >
              <CardContent className="p-5">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={(e) =>
                    onRevokeAccess(acc.provider_id || acc.account_id, e)
                  }
                  disabled={revokingProviderId === acc.provider_id}
                  className="absolute top-2 right-2 h-8 w-8 text-slate-500 hover:text-red-400 hover:bg-red-400/10 z-10"
                >
                  {revokingProviderId === acc.provider_id ? (
                    <Loader2 size={14} className="animate-spin" />
                  ) : (
                    <Unlink size={14} />
                  )}
                </Button>

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
                  <div className="mt-4 flex items-center justify-center gap-2 py-2 bg-red-500/10 text-red-400 rounded-md text-xs font-bold">
                    <AlertTriangle size={14} /> Reconnect Required
                  </div>
                )}
              </CardContent>
            </Card>
          );
        })}

      <Button
        variant="outline"
        onClick={onLinkBank}
        className="w-full mt-2 h-14 border-dashed border-slate-700 bg-transparent text-slate-400 hover:text-[#00FFAA] hover:border-[#00FFAA]/50 hover:bg-[#00FFAA]/5 gap-2"
      >
        <Plus size={18} />
        <span className="text-xs font-bold tracking-widest uppercase">
          Link Bank
        </span>
      </Button>
    </aside>
  );
};
