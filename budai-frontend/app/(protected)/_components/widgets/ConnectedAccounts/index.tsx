"use client";

import React, { useState, useEffect } from "react";
import { Card, Button, Avatar, Badge, toast } from "@heroui/react";
import { CreditCard, Plus } from "lucide-react";
import { useBudAI } from "@/app/context/AppContext";
import { WidgetContext } from "@/app/(protected)/home/DashboardClient";
import { useRouter } from "next/navigation";
import { Account } from "@/types";
import { cn } from "@/lib/utils";
import { apiFetch } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";

interface GroupedAccounts {
  bankName: string;
  logoUrl?: string;
  accounts: Account[];
  isExpired: boolean;
  isHardRevoked: boolean;
  bankUuid?: string;
}

export default function ConnectedAccountsWidgetClient() {
  const router = useRouter();
  const { onRemove } = React.useContext(WidgetContext);
  const { accounts } = useBudAI();
  const [isExpanded, setIsExpanded] = useState(false);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [isReauthenticating, setIsReauthenticating] = useState(false);

  useEffect(() => {
    setIsLoading(false);
  }, [accounts]);

  const handleReauth = async (bankUuid?: string) => {
    if (!bankUuid) return;
    setIsReauthenticating(true);
    try {
      const res = await apiFetch(`/api/banks/${bankUuid}/reauth`, { method: "POST" }, true);
      if (res.ok) {
        const data = await res.json() as any;
        if (data.auth_url) {
          router.push(data.auth_url); 
        }
      } else {
        toast.danger("Failed to initialize secure connection");
      }
    } catch (error) {
      toast.danger("Failed to initialize secure connection");
    } finally {
      setIsReauthenticating(false);
    }
  };

  const handleConnectAccount = () => {
    setIsConnecting(true);
    router.push("/connections");
  };

  const formatSortCode = (sortCode?: string) => {
    if (!sortCode) return "";
    const clean = sortCode.replace(/[^a-zA-Z0-9]/g, "");
    return clean.replace(/(.{2})(?=.)/g, "$1-");
  };

  const groupedAccounts = React.useMemo(() => {
    if (!accounts) return [];

    const groups = accounts.reduce((acc, account) => {
      let bankName = account.provider_name || "Bank Account";
      let logoUrl = (account as typeof account & { logo_url?: string }).logo_url;

      if (account.provider_name === "truelayer") {
        bankName = account.bank_name || "TrueLayer Bank";
      }

      if (!acc[bankName]) {
        acc[bankName] = {
          bankName,
          logoUrl,
          accounts: [],
          isExpired: false,
          isHardRevoked: false,
          bankUuid: account.bank_uuid,
        };
      }
      acc[bankName].accounts.push(account);
      if (account.consent_status === "200-401" || account.consent_status === "200-403") {
        acc[bankName].isExpired = true;
        if (account.consent_status === "200-403") {
          acc[bankName].isHardRevoked = true;
        }
        if (account.bank_uuid) acc[bankName].bankUuid = account.bank_uuid;
      }
      return acc;
    }, {} as Record<string, GroupedAccounts>);

    return Object.values(groups);
  }, [accounts]);

  const displayGroups = groupedAccounts;

  if (isLoading) {
    return (
      <Card className="w-full h-full liquid-glass rounded-xl p-8 flex flex-col animate-pulse">
        <div className="h-4 w-32 bg-foreground/10 rounded mb-8"></div>
        <div className="flex-1 rounded-xl bg-foreground/5 border border-foreground/10"></div>
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

  return (
    <Card className="w-full h-full liquid-glass rounded-xl flex flex-col overflow-hidden">
      <Card.Header className="flex justify-between items-center p-8 pb-4 shrink-0 z-20">
        <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0 flex items-center gap-2">
          Your Bank Accounts
          <Badge color="default" variant="primary" size="sm">{accounts.length}</Badge>
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
        <div className="w-full flex-1 overflow-y-auto overflow-x-hidden min-h-0 scrollbar-hide relative pb-10">
          <div
            className="relative w-full transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] mt-2"
            style={{
              height: isExpanded
                ? `${displayGroups.length * 360 + 20}px`
                : "100%",
            }}
          >
            {displayGroups.map((group, idx) => {
              const bankName = group.bankName;
              const logoUrl = group.logoUrl;

              return (
                <div
                  key={bankName}
                  onClick={(e) => {
                    if (group.isExpired) {
                      e.stopPropagation();
                      handleReauth(group.bankUuid);
                      return;
                    }
                    if (displayGroups.length > 1) {
                      setIsExpanded(!isExpanded);
                    }
                  }}
                  className={cn(
                    "group absolute w-full rounded-xl p-8 overflow-hidden transition-all duration-700 ease-[cubic-bezier(0.16,1,0.3,1)] origin-top flex flex-col justify-between",
                    group.isExpired ? "bg-black backdrop-blur-md border-[0.5px] border-red-500/50 shadow-[inset_0_0_50px_rgba(239,68,68,0.1)] cursor-pointer hover:bg-black/90" : "bg-black bg-gradient-to-br from-indigo-500/10 to-purple-500/10 backdrop-blur-xl border-[0.5px] border-indigo-500/30 group-hover:border-indigo-500/50 shadow-[inset_0_0_50px_rgba(99,102,241,0.05)]",
                    !group.isExpired && displayGroups.length > 1 && "cursor-pointer",
                    !group.isExpired && displayGroups.length > 1 && !isExpanded && "hover:brightness-125"
                  )}
                  style={{
                    height: isExpanded ? '338px' : '100%',
                    top: isExpanded ? `${idx * 360}px` : `${idx * 36}px`,
                    transform: isExpanded
                      ? `scale(1)`
                      : `scale(${1 - idx * 0.05})`,
                    zIndex: 50 - idx,
                    opacity: isExpanded ? 1 : (idx === 0 ? 0.85 : 1 - idx * 0.15),
                  }}
                >
                  <div className="flex justify-between items-start z-10">
                    <div className="flex flex-row items-center gap-8 flex-wrap">
                      {group.isExpired ? (
                        <div className="flex flex-col gap-2">
                          <div className="flex items-center gap-2 mb-2">
                             <div className="bg-red-500/20 text-red-500 border border-red-500/50 px-3 py-1 rounded-full text-[9px] font-black uppercase tracking-widest flex items-center gap-2">
                               <span className="animate-pulse">⚠️</span> {group.isHardRevoked ? "Access Revoked" : "Connection Expired"}
                             </div>
                          </div>
                          <span className="text-[12px] font-mono text-red-500/80 uppercase tracking-widest mt-1">
                             {isReauthenticating ? "Reconnecting..." : "Tap to Reconnect"}
                          </span>
                        </div>
                      ) : (
                        group.accounts.map(acc => (
                          <div key={acc.account_id} className="flex flex-col gap-2">
                            <span className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.3em]">
                              {acc.currency || "GBP"} Balance
                            </span>
                            <h2 className="text-4xl font-normal text-foreground tracking-tighter mt-1 font-mono">
                              {new Intl.NumberFormat("en-GB", {
                                style: "currency",
                                currency: acc.currency || "GBP",
                                minimumFractionDigits: 2,
                                maximumFractionDigits: 2,
                              }).format(acc.balance ?? 0)}
                            </h2>
                          </div>
                        ))
                      )}
                    </div>
                    <Avatar
                      variant="soft"
                      className="w-14 h-14 bg-foreground p-2 shadow-xl text-background font-black text-xl rounded-xl group-hover:scale-105 transition-all flex justify-center items-center shrink-0 ml-4"
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
                        <span>*{group.accounts[0]?.account_number?.slice(-4) || "****"}</span>
                        <span className="opacity-20">|</span>
                        <span>{formatSortCode(group.accounts[0]?.sort_code)}</span>
                        {group.accounts.length > 1 && (
                          <span className="ml-2 px-1.5 py-0.5 rounded bg-white/10 text-[8px]">+{group.accounts.length - 1} MORE</span>
                        )}
                      </div>
                    </div>
                    <div className="flex gap-1 opacity-20 group-hover:opacity-40 transition-opacity">
                      <div className="w-8 h-8 rounded-full border border-white/40"></div>
                      <div className="w-8 h-8 rounded-full border border-white/40 -ml-4 bg-white/10"></div>
                    </div>
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
