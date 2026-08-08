"use client";

import React, { useState } from "react";
import { ChevronDown, ChevronRight, CreditCard, Plus } from "lucide-react";
import { cn } from "@/lib/utils";
import { useBudAI } from "@/app/context/AppContext";
import { Card, Button, Avatar, Skeleton, Badge } from "@heroui/react";
import { apiFetch } from "@/lib/api";
import { useRouter } from "next/navigation";
import { WidgetContext } from "../../../home/DashboardClient";

export default function PortfolioCardWidget() {
  const router = useRouter();
  const { onRemove } = React.useContext(WidgetContext);
  const { accounts } = useBudAI();
  const [isExpanded, setIsExpanded] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  React.useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 800);

    if (accounts.length > 0) {
      setIsLoading(false);
      clearTimeout(timer);
    }

    return () => clearTimeout(timer);
  }, [accounts]);

  const displayAccounts = accounts.slice(0, 4);

  const getGradient = (index: number, expanded: boolean) => {
    const gradients = [
      "bg-gradient-to-br from-indigo-500/10 to-purple-500/10 border-indigo-500/30",
      "bg-gradient-to-br from-emerald-500/10 to-teal-500/10 border-emerald-500/30",
      "bg-gradient-to-br from-orange-500/10 to-rose-500/10 border-orange-500/30",
      "bg-gradient-to-br from-blue-500/10 to-cyan-500/10 border-blue-500/30",
    ];
    return `${gradients[index % gradients.length]} border-[0.5px] shadow-inner backdrop-blur-xl`;
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
        const data = await res.json() as any;
        if (data.auth_url) {
          router.push(data.auth_url);
        }
      }
    } catch (error) {
      console.error(error);
    } finally {
      setIsConnecting(false);
    }
  };

  if (isLoading) {
    return (
      <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
        <div className="flex justify-between items-center p-8 mb-10">
          <Skeleton
            className="h-6 w-32 rounded-lg bg-white/5"
            animationType="shimmer"
          />
          <Skeleton
            className="h-6 w-6 rounded-md bg-white/5"
            animationType="shimmer"
          />
        </div>
        <div className="space-y-6 px-8 pb-8">
          <div className="flex justify-between items-start">
            <div className="space-y-4">
              <Skeleton
                className="h-2 w-20 rounded bg-white/5"
                animationType="shimmer"
              />
              <Skeleton
                className="h-10 w-40 rounded-xl bg-white/5"
                animationType="shimmer"
              />
            </div>
            <Skeleton
              className="h-14 w-14 rounded-xl bg-white/5"
              animationType="shimmer"
            />
          </div>
        </div>
      </Card>
    );
  }

  if (!accounts || accounts.length === 0) {
    return (
      <Card className="w-full h-full liquid-glass rounded-xl p-10 flex flex-col justify-center items-center text-center">
        <CreditCard className="w-10 h-10 text-foreground/20 mb-4" />
        <span className="text-[10px] font-black text-foreground/40 uppercase tracking-[0.3em] mb-8">
          No bank connected
        </span>
        <Button
          variant="primary"
          onPress={handleConnectAccount}
          isPending={isConnecting}
          className="font-black text-[10px] uppercase tracking-widest px-8 h-12 rounded-lg cursor-pointer bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 shadow-lg"
        >
          Connect Bank
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
    const balance = acc.balance ?? 0;

    return (
      <Card className="w-full h-full liquid-glass rounded-xl flex flex-col overflow-hidden">
        <Card.Header className="flex justify-between items-center p-8 pb-4 shrink-0 z-20">
          <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
            Your Bank Accounts
          </h3>
          <div className="flex items-center gap-2">
            <Button
              isIconOnly
              variant="ghost"
              onPress={handleConnectAccount}
              isPending={isConnecting}
              className="flex items-center justify-center p-0 text-foreground/30 hover:text-primary transition-colors border-none bg-transparent"
            >
              <Plus size={16} />
            </Button>
          </div>
        </Card.Header>

        <Card.Content className="relative flex-1 w-full p-8 pt-0 flex flex-col h-full overflow-hidden">
          <div className="relative flex-1 w-full rounded-xl p-8 overflow-hidden flex flex-col justify-between bg-gradient-to-br from-indigo-500/10 to-purple-500/10 backdrop-blur-xl border-[0.5px] border-indigo-500/30 group hover:border-indigo-500/50 transition-all duration-500 shadow-inner">
            <div className="flex justify-between items-start z-10">
              <div className="flex flex-col gap-2">
                <span className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.3em]">
                  Live Account balance
                </span>
                <h2 className="text-5xl font-normal text-foreground tracking-tighter mt-1 font-mono">
                  {new Intl.NumberFormat("en-GB", {
                    style: "currency",
                    currency: acc.currency || "GBP",
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  }).format(balance)}
                </h2>
              </div>
              <Avatar
                variant="soft"
                className="w-14 h-14 bg-foreground p-2 shadow-xl text-background font-black text-xl rounded-xl group-hover:scale-105 transition-all flex justify-center items-center"
              >
                {logoUrl && <Avatar.Image src={logoUrl} alt={bankName} />}
                <Avatar.Fallback>{bankName.charAt(0)}</Avatar.Fallback>
              </Avatar>
            </div>

            <div className="flex justify-between items-end z-10 mt-auto pt-8">
              <div className="flex flex-col gap-1">
                <span className="text-foreground font-black text-lg tracking-tighter uppercase italic">
                  {bankName}
                </span>
                <div className="flex items-center gap-4 text-foreground/30 text-[10px] font-bold tracking-[0.2em] font-mono">
                  <span>*{accountNumber.slice(-4)}</span>
                  <span className="opacity-20">|</span>
                  <span>{sortCode}</span>
                </div>
              </div>
              <div className="flex gap-1 opacity-20 group-hover:opacity-40 transition-opacity">
                <div className="w-8 h-8 rounded-full border border-white/40"></div>
                <div className="w-8 h-8 rounded-full border border-white/40 -ml-4 bg-white/10"></div>
              </div>
            </div>
          </div>
        </Card.Content>
      </Card>
    );
  }

  return (
    <Card className="w-full h-full liquid-glass rounded-xl flex flex-col overflow-hidden">
      <Card.Header className="flex justify-between items-center p-8 pb-4 shrink-0 z-20">
        <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0 flex items-center gap-2">
          Your Accounts
          <Badge color="default" variant="primary" size="sm">{accounts.length}</Badge>
        </h3>
        <div className="flex items-center gap-4">

          <Button
            isIconOnly
            variant="ghost"
            onPress={handleConnectAccount}
            isPending={isConnecting}
            className="text-foreground/30 hover:text-primary transition-colors border-none bg-transparent"
          >
            <Plus size={16} />
          </Button>
        </div>
      </Card.Header>

      <Card.Content className="relative flex-1 w-full p-0 overflow-hidden flex flex-col">
        <div className="w-full flex-1 overflow-y-auto overflow-x-hidden px-8 pb-8 min-h-0 scrollbar-hide">
          <div
            className="relative w-full transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] mt-2"
            style={{
              height: isExpanded
                ? `${displayAccounts.length * 190 + 20}px`
                : "230px",
            }}
          >
            {displayAccounts.map((acc, idx) => {
              const sortCode = formatSortCode(acc.sort_code);
              const accountNumber = acc.account_number || "****";
              const bankName = acc.bank_name || "Bank";
              const logoUrl = (acc as typeof acc & { logo_url?: string })
                .logo_url;
              const balance = acc.balance ?? 0;

              return (
                <div
                  key={acc.account_id}
                  onClick={() => setIsExpanded(!isExpanded)}
                  className={cn(
                    "group absolute w-full h-44 rounded-xl p-8 overflow-hidden transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] origin-top flex flex-col justify-center cursor-pointer hover:brightness-125 border-none",
                    getGradient(idx, isExpanded),
                  )}
                  style={{
                    top: isExpanded ? `${idx * 190}px` : `${idx * 36}px`,
                    transform: isExpanded
                      ? `scale(1)`
                      : `scale(${1 - idx * 0.05})`,
                    zIndex: 50 - idx,
                    opacity: isExpanded ? 1 : (idx === 0 ? 0.85 : 1 - idx * 0.15),
                  }}
                >
                  <div className="absolute right-8 top-8 opacity-90 flex items-center gap-4 z-10">
                    <span className="text-foreground font-black text-lg tracking-tighter uppercase italic">
                      {bankName}
                    </span>
                    <Avatar
                      variant="soft"
                      className="w-12 h-12 bg-foreground p-1.5 shadow-xl text-background font-black text-xl rounded-xl"
                    >
                      {logoUrl && <Avatar.Image src={logoUrl} alt={bankName} />}
                      <Avatar.Fallback>{bankName.charAt(0)}</Avatar.Fallback>
                    </Avatar>
                  </div>

                  <div className="absolute right-8 bottom-8 flex gap-1 opacity-20 z-10 pointer-events-none">
                    <div className="w-8 h-8 rounded-full border border-white/40"></div>
                    <div className="w-8 h-8 rounded-full border border-white/40 -ml-4 bg-white/10"></div>
                  </div>

                  <span className="text-[9px] font-black text-foreground/40 uppercase tracking-[0.3em] mb-1 z-10">
                    Current Balance
                  </span>
                  <h2 className="text-3xl font-normal text-foreground tracking-tighter z-10 font-mono">
                    {new Intl.NumberFormat("en-GB", {
                      style: "currency",
                      currency: acc.currency || "GBP",
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    }).format(balance)}
                  </h2>

                  <div className="mt-8 flex justify-between items-center text-foreground/30 text-[10px] font-bold font-mono tracking-[0.2em] z-10">
                    <span className="flex items-center gap-2">
                      <span>*{accountNumber.slice(-4)}</span>
                      <span className="opacity-40">|</span>
                      <span>{sortCode}</span>
                    </span>
                    {idx === 0 && displayAccounts.length > 1 && (
                      <Button
                        variant="ghost"
                        size="sm"
                        onClick={(e) => e.stopPropagation()}
                        onPress={() => setIsExpanded(!isExpanded)}
                        className="text-[9px] font-black text-foreground/30 hover:text-foreground uppercase tracking-widest flex items-center gap-1 bg-transparent border-none cursor-pointer transition-colors"
                      />
                    )}
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
