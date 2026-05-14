import React, { useState } from "react";
import {
  ChevronDown,
  ChevronRight,
  CreditCard,
  TrendingUp,
  Plus,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { useBudAI } from "@/app/context/AppContext";
import { Card, Button, Avatar } from "@heroui/react";
import { apiFetch } from "@/lib/api";

export default function PortfolioCardWidget() {
  const { accounts } = useBudAI();
  const [isExpanded, setIsExpanded] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);

  const displayAccounts = accounts.slice(0, 4);

  const getGradient = (index: number, expanded: boolean) => {
    if (expanded) {
      const translucentGradients = [
        "bg-neon-cyan/10 border border-neon-cyan/50 shadow-[0_0_20px_rgba(0,229,255,0.1)]",
        "bg-deep-pink/10 border border-deep-pink/50 shadow-[0_0_20px_rgba(255,51,102,0.1)]",
        "bg-[#B900FF]/10 border border-[#B900FF]/50 shadow-[0_0_20px_rgba(185,0,255,0.1)]",
        "bg-[#39FF14]/10 border border-[#39FF14]/50 shadow-[0_0_20px_rgba(57,255,20,0.1)]",
      ];
      return translucentGradients[index % translucentGradients.length];
    } else {
      const opaqueGradients = [
        "bg-obsidian border border-neon-cyan/40 shadow-[0_0_15px_rgba(0,229,255,0.05)]",
        "bg-obsidian border border-deep-pink/40 shadow-[0_0_15px_rgba(255,51,102,0.05)]",
        "bg-obsidian border border-[#B900FF]/40 shadow-[0_0_15px_rgba(185,0,255,0.05)]",
        "bg-obsidian border border-[#39FF14]/40 shadow-[0_0_15px_rgba(57,255,20,0.05)]",
      ];
      return opaqueGradients[index % opaqueGradients.length];
    }
  };

  const formatSortCode = (sc: string | undefined) => {
    if (!sc) return "00-00-00";
    const cleaned = sc.replace(/\D/g, "");
    if (cleaned.length === 6) {
      return `${cleaned.slice(0, 2)}-${cleaned.slice(2, 4)}-${cleaned.slice(4, 6)}`;
    }
    return sc;
  };

  const handleConnectAccount = async () => {
    setIsConnecting(true);
    try {
      const res = await apiFetch("/api/auth/truelayer/status", {}, true);
      if (res.ok) {
        const data = await res.json();
        if (data.auth_url) {
          window.location.href = data.auth_url;
        }
      }
    } catch (error) {
      console.error(error);
    } finally {
      setIsConnecting(false);
    }
  };

  if (!accounts || accounts.length === 0) {
    return (
      <Card className="w-full h-full obsidian-glass p-6 flex flex-col shadow-2xl justify-center items-center font-geist">
        <CreditCard className="w-12 h-12 text-[#5E6272] mb-3 opacity-50" />
        <span className="text-[#8B8E98] text-sm font-medium mb-6">
          No Active Accounts
        </span>
        <Button
          variant="primary"
          onPress={handleConnectAccount}
          isPending={isConnecting}
          className="bg-transparent border border-neon-cyan/50 text-neon-cyan hover:bg-neon-cyan/10 font-bold px-6 h-11 rounded-xl transition-colors cursor-pointer"
        >
          Connect Account
        </Button>
      </Card>
    );
  }

  if (accounts.length === 1) {
    const acc = accounts[0];
    const sortCode = formatSortCode(acc.sort_code);
    const accountNumber = acc.account_number || "****";
    const bankName = acc.bank_name || "Bank";
    const logoUrl = (acc as typeof acc & { logo_url?: string }).logo_url;
    const balance = acc.account_balance ?? acc.balance ?? 0;

    return (
      <Card className="w-full h-full obsidian-glass flex flex-col shadow-2xl overflow-hidden font-geist">
        <Card.Header className="flex justify-between items-center p-6 pb-4 shrink-0 z-20">
          <Card.Title className="text-white font-bold text-2xl tracking-tight">
            Active Account
          </Card.Title>
          <Button
            isIconOnly
            variant="ghost"
            onPress={handleConnectAccount}
            isPending={isConnecting}
            className="flex items-center justify-center p-0 text-[#8B8E98] hover:text-neon-cyan hover:bg-neon-cyan/10 min-w-8 w-8 h-8 rounded-lg cursor-pointer transition-colors border-none bg-transparent"
          >
            <Plus size={18} />
          </Button>
        </Card.Header>

        <Card.Content className="relative flex-1 w-full p-6 pt-0 flex flex-col h-full overflow-hidden">
          <div className="relative flex-1 w-full rounded-2xl p-8 shadow-2xl overflow-hidden flex flex-col justify-between bg-[#0A0A0A] border border-neon-cyan/40 group hover:border-neon-cyan/70 transition-colors">
            <div className="absolute inset-0 bg-black/40 pointer-events-none z-0"></div>
            <div className="absolute -right-32 -top-32 w-96 h-96 bg-neon-cyan/10 rounded-full blur-[100px] pointer-events-none z-0"></div>
            <div className="absolute -left-32 -bottom-32 w-96 h-96 bg-deep-pink/5 rounded-full blur-[100px] pointer-events-none z-0"></div>

            <svg
              className="absolute bottom-0 left-0 w-full h-40 opacity-40 pointer-events-none z-0"
              preserveAspectRatio="none"
              viewBox="0 0 100 100"
            >
              <path
                d="M0,100 L0,50 Q15,80 35,40 T70,60 T100,30 L100,100 Z"
                fill="url(#sparkline-fill)"
              />
              <defs>
                <linearGradient id="sparkline-fill" x1="0" y1="0" x2="0" y2="1">
                  <stop offset="0%" stopColor="#00E5FF" stopOpacity="0.4" />
                  <stop offset="100%" stopColor="#00E5FF" stopOpacity="0" />
                </linearGradient>
              </defs>
            </svg>
            <svg
              className="absolute bottom-0 left-0 w-full h-40 opacity-70 pointer-events-none z-0 drop-shadow-[0_0_8px_rgba(0,229,255,0.8)]"
              preserveAspectRatio="none"
              viewBox="0 0 100 100"
            >
              <path
                d="M0,50 Q15,80 35,40 T70,60 T100,30"
                fill="none"
                stroke="#00E5FF"
                strokeWidth="1.5"
              />
            </svg>

            <div className="flex justify-between items-start z-10">
              <div className="flex flex-col gap-1">
                <span className="text-[#8B8E98] text-xs font-bold tracking-widest uppercase">
                  Current Balance
                </span>
                <h2 className="text-5xl font-black text-white tracking-tighter mt-1 drop-shadow-md">
                  £{" "}
                  {balance.toLocaleString(undefined, {
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  })}
                </h2>
                <div className="flex items-center gap-2 mt-3 px-3 py-1.5 rounded-full bg-brand-green/10 border border-brand-green/20 text-brand-green text-xs font-bold tracking-wide w-fit">
                  <TrendingUp size={14} strokeWidth={3} />
                  <span>+3.2% vs last month</span>
                </div>
              </div>
              <Avatar className="w-16 h-16 bg-white p-2 shadow-[0_0_25px_rgba(0,229,255,0.15)] text-[#08090D] font-black text-2xl rounded-2xl group-hover:scale-105 transition-transform">
                {logoUrl && <Avatar.Image src={logoUrl} alt={bankName} />}
                <Avatar.Fallback>{bankName.charAt(0)}</Avatar.Fallback>
              </Avatar>
            </div>

            <div className="flex justify-between items-end z-10 mt-auto pt-8">
              <div className="flex flex-col gap-1">
                <span className="text-white font-bold text-xl tracking-tight">
                  {bankName}
                </span>
                <div className="flex items-center gap-3 text-[#8B8E98] text-sm font-semibold tracking-widest font-mono">
                  <span>•••• {accountNumber.slice(-4)}</span>
                  <span className="opacity-30">|</span>
                  <span>{sortCode}</span>
                </div>
              </div>
              <div className="flex gap-1 opacity-30 group-hover:opacity-50 transition-opacity">
                <div className="w-10 h-10 rounded-full border-2 border-white"></div>
                <div className="w-10 h-10 rounded-full border-2 border-white -ml-5 backdrop-blur-md bg-white/10"></div>
              </div>
            </div>
          </div>
        </Card.Content>
      </Card>
    );
  }

  return (
    <Card className="w-full h-full obsidian-glass flex flex-col shadow-2xl overflow-hidden font-geist">
      <Card.Header className="flex justify-between items-center p-6 pb-4 shrink-0 z-20">
        <Card.Title className="text-white font-bold text-2xl tracking-tight">
          Active Accounts
        </Card.Title>
        <div className="flex items-center gap-2">
          <Button
            variant="ghost"
            size="sm"
            onPress={() => setIsExpanded(!isExpanded)}
            className="text-[#8B8E98] hover:text-white font-medium flex items-center gap-1 bg-transparent border-none cursor-pointer transition-colors"
          >
            {isExpanded ? "Collapse" : "Show All"}
            {isExpanded ? (
              <ChevronDown size={14} />
            ) : (
              <ChevronRight size={14} />
            )}
          </Button>
          <Button
            isIconOnly
            variant="ghost"
            onPress={handleConnectAccount}
            isPending={isConnecting}
            className="text-[#8B8E98] hover:text-neon-cyan hover:bg-neon-cyan/10 min-w-8 w-8 h-8 rounded-lg cursor-pointer transition-colors border-none bg-transparent"
          >
            <Plus size={18} />
          </Button>
        </div>
      </Card.Header>

      <Card.Content className="relative flex-1 w-full p-0 overflow-hidden flex flex-col">
        <div className="w-full flex-1 overflow-y-auto overflow-x-hidden px-6 pb-6 min-h-0 [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
          <div
            className="relative w-full transition-all duration-500 ease-in-out mt-2"
            style={{
              height: isExpanded
                ? `${displayAccounts.length * 210 + 20}px`
                : "220px",
            }}
          >
            {displayAccounts.map((acc, idx) => {
              const sortCode = formatSortCode(acc.sort_code);
              const accountNumber = acc.account_number || "****";
              const bankName = acc.bank_name || "Bank";
              const logoUrl = (acc as typeof acc & { logo_url?: string })
                .logo_url;
              const balance = acc.account_balance ?? acc.balance ?? 0;

              return (
                <div
                  key={acc.account_id}
                  className={cn(
                    "absolute w-full h-48 rounded-3xl p-6 shadow-2xl overflow-hidden transition-all duration-500 ease-in-out flex flex-col justify-center cursor-pointer hover:brightness-110",
                    getGradient(idx, isExpanded),
                  )}
                  style={{
                    top: isExpanded ? `${idx * 210}px` : `${idx * 16}px`,
                    transform: isExpanded
                      ? `scale(1)`
                      : `scale(${1 - idx * 0.05})`,
                    zIndex: 50 - idx,
                    opacity: isExpanded ? 1 : 1 - idx * 0.15,
                  }}
                >
                  <div className="absolute inset-0 bg-black/20 mix-blend-multiply pointer-events-none"></div>
                  <div className="absolute -right-10 -top-10 w-48 h-48 bg-white/5 rounded-full blur-3xl transition-colors pointer-events-none"></div>

                  <div className="absolute right-6 top-6 opacity-90 flex items-center gap-4 z-10">
                    <span className="text-white font-bold text-xl tracking-tight">
                      {bankName}
                    </span>
                    <Avatar className="w-14 h-14 bg-white p-1.5 shadow-[0_0_15px_rgba(255,255,255,0.2)] text-[#08090D] font-black text-xl rounded-2xl">
                      {logoUrl && <Avatar.Image src={logoUrl} alt={bankName} />}
                      <Avatar.Fallback>{bankName.charAt(0)}</Avatar.Fallback>
                    </Avatar>
                  </div>

                  <div className="absolute right-6 bottom-6 flex gap-1 opacity-20 z-10 pointer-events-none">
                    <div className="w-10 h-10 rounded-full border-2 border-white/80"></div>
                    <div className="w-10 h-10 rounded-full border-2 border-white/80 -ml-5 backdrop-blur-md bg-white/10"></div>
                  </div>

                  <span className="text-[#8B8E98] text-xs font-bold uppercase tracking-widest mb-1 z-10">
                    Account Balance
                  </span>
                  <h2 className="text-3xl font-black text-white tracking-tighter z-10">
                    £{" "}
                    {balance.toLocaleString(undefined, {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                  </h2>

                  <div className="mt-8 flex justify-between items-center text-[#8B8E98] text-sm font-semibold font-mono tracking-widest z-10">
                    <span className="flex items-center gap-2">
                      <span>•••• {accountNumber.slice(-4)}</span>
                      <span className="opacity-40">|</span>
                      <span>{sortCode}</span>
                    </span>
                  </div>
                </div>
              );
            })}
          </div>
        </div>
      </Card.Content>
    </Card>
  );
}
