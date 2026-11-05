"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Avatar, Chip, Button, Dropdown } from "@heroui/react";
import { useRouter } from "next/navigation";
import Image from "next/image";
import NextLink from "next/link";
import {
  Search,
  LogOut,
  ChevronDown,
  Building2,
  CheckCircle2,
  AlertCircle,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { apiFetch, clearSession } from "@/lib/api";
import { useSearch } from "@/app/context/SearchContext";
import { useBudAI } from "@/app/context/AppContext";
import { useQuery } from "@tanstack/react-query";
import { formatCurrency, getCurrencySymbol } from "@/lib/currency";

export const TopNavbar = () => {
  const router = useRouter();
  const { openSearch } = useSearch();
  const { accounts, activeAccountId, setActiveAccountId } = useBudAI();
  const [persona, setPersona] = useState<string | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [email, setEmail] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPersona() {
      try {
        const res = await apiFetch(`/api/auth/me`, {}, true);
        if (res.ok) {
          const json = (await res.json()) as {
            persona: string;
            name: string;
            email: string;
          };
          setPersona(json.persona);
          setName(json.name);
          setEmail(json.email);
        }
      } catch (err) {
        console.error("Failed to fetch user profile", err);
      }
    }
    fetchPersona();
  }, []);

  const { data: buckets = [] } = useQuery({
    queryKey: ["buckets"],
    queryFn: async () => {
      const res = await apiFetch("/api/buckets/");
      if (res.ok) {
        const data = (await res.json()) as { buckets?: any[] };
        return data.buckets || [];
      }
      return [];
    },
    refetchInterval: 15000,
  });

  const currencySummaries = useMemo(() => {
    const relevantAccounts =
      !activeAccountId || activeAccountId === "ALL"
        ? accounts
        : accounts.filter((a) => a.account_id === activeAccountId);

    const currencyMap: Record<
      string,
      { physical: number; assigned: number }
    > = {};

    relevantAccounts.forEach((acc) => {
      const cur = (acc.currency || "GBP").toUpperCase();
      if (!currencyMap[cur]) currencyMap[cur] = { physical: 0, assigned: 0 };
      currencyMap[cur].physical += acc.balance || 0;
    });

    buckets.forEach((b: any) => {
      const cur = (b.currency || "GBP").toUpperCase();
      if (!currencyMap[cur]) currencyMap[cur] = { physical: 0, assigned: 0 };
      currencyMap[cur].assigned += b.cached_balance || 0;
    });

    return Object.entries(currencyMap).map(
      ([currency, { physical, assigned }]) => {
        const diff = physical - assigned;
        return {
          currency,
          physical,
          assigned,
          diff,
          isBalanced: Math.abs(diff) < 0.01,
        };
      },
    );
  }, [accounts, activeAccountId, buckets]);

  const activeAccount = useMemo(() => {
    if (!activeAccountId || activeAccountId === "ALL") return null;
    return accounts.find((a) => a.account_id === activeAccountId) || null;
  }, [accounts, activeAccountId]);

  const handleLogout = () => {
    clearSession();
    router.push("/login");
  };

  return (
    <header className="relative w-full h-16 md:h-20 bg-background/85 backdrop-blur-xl border-b border-border/40 flex justify-between items-center px-4 sm:px-6 lg:px-8 xl:px-12 shrink-0 shadow-sm z-40 transition-colors">
      <div className="flex items-center gap-3.5">
        <NextLink href="/home" className="flex items-center shrink-0">
          <Image
            src="/FullLogo.jpg"
            alt="BudAI"
            width={82}
            height={26}
            className="rounded-sm object-contain"
            priority
          />
        </NextLink>

        <div className="h-5 w-[1px] bg-border/50 hidden sm:block" />

        <Dropdown>
          <Button
            variant="secondary"
            className="flex items-center gap-2 px-3 py-1.5 h-auto rounded-xl bg-foreground/5 hover:bg-foreground/10 border border-border/40 text-foreground text-sm font-semibold transition-all shadow-none outline-none cursor-pointer"
          >
            <Building2 size={15} className="text-foreground/70 shrink-0" />
            <span className="truncate max-w-[160px] md:max-w-[220px]">
              {activeAccount
                ? activeAccount.bank_name || activeAccount.provider_name || "Account"
                : "All Connected Accounts"}
            </span>
            <ChevronDown size={14} className="text-foreground/50 shrink-0" />
          </Button>
          <Dropdown.Popover className="bg-popover/95 backdrop-blur-2xl border border-border shadow-2xl rounded-xl w-72 p-1.5">
            <Dropdown.Menu
              onAction={(key) => {
                setActiveAccountId(key === "ALL" ? "ALL" : String(key));
              }}
            >
              <Dropdown.Item
                id="ALL"
                textValue="All Connected Accounts"
                className={cn(
                  "px-3 py-2.5 rounded-lg cursor-pointer text-sm font-medium hover:bg-foreground/5",
                  (!activeAccountId || activeAccountId === "ALL") &&
                    "text-primary font-bold bg-primary/10",
                )}
              >
                All Connected Accounts ({accounts.length})
              </Dropdown.Item>
              {accounts.map((acc) => (
                <Dropdown.Item
                  key={acc.account_id}
                  id={acc.account_id}
                  textValue={acc.bank_name || "Account"}
                  className={cn(
                    "px-3 py-2.5 rounded-lg cursor-pointer text-sm font-medium hover:bg-foreground/5 flex items-center justify-between",
                    activeAccountId === acc.account_id &&
                      "text-primary font-bold bg-primary/10",
                  )}
                >
                  <div className="flex flex-col">
                    <span className="font-semibold">
                      {acc.bank_name || acc.provider_name || "Bank Account"}
                    </span>
                    <span className="text-xs text-muted-foreground font-mono">
                      {formatCurrency(acc.balance || 0, acc.currency || "GBP")}
                    </span>
                  </div>
                </Dropdown.Item>
              ))}
            </Dropdown.Menu>
          </Dropdown.Popover>
        </Dropdown>
      </div>

      <div className="hidden lg:flex items-center gap-2.5 overflow-x-auto py-1 scrollbar-hide">
        {currencySummaries.length === 0 ? (
          <div className="flex items-center gap-2 px-3.5 py-1.5 rounded-full bg-foreground/5 border border-border/40 text-xs text-muted-foreground">
            <span>No bank accounts linked</span>
          </div>
        ) : (
          currencySummaries.map((summary) => {
            const sym = getCurrencySymbol(summary.currency);
            return (
              <div
                key={summary.currency}
                className={cn(
                  "flex items-center gap-2.5 px-4 py-1.5 rounded-full text-sm font-semibold border transition-all shadow-xs",
                  summary.isBalanced
                    ? "bg-emerald-500/10 border-emerald-500/25 text-emerald-600 dark:text-emerald-400"
                    : summary.diff > 0
                    ? "bg-amber-500/10 border-amber-500/25 text-amber-600 dark:text-amber-400"
                    : "bg-red-500/10 border-red-500/25 text-red-600 dark:text-red-400",
                )}
              >
                {summary.isBalanced ? (
                  <CheckCircle2 size={15} className="shrink-0" />
                ) : (
                  <AlertCircle size={15} className="shrink-0" />
                )}
                <span className="font-bold">
                  {formatCurrency(summary.physical, summary.currency)} Bank
                </span>
                <span className="text-xs opacity-50">·</span>
                <span className="text-xs md:text-sm font-semibold">
                  {summary.isBalanced
                    ? "All Balanced"
                    : summary.diff > 0
                    ? `+${sym}${Math.abs(summary.diff).toFixed(2)} Unassigned`
                    : `-${sym}${Math.abs(summary.diff).toFixed(2)} Overallocated`}
                </span>
              </div>
            );
          })
        )}
      </div>

      <div className="flex items-center gap-3">
        <Button
          isIconOnly
          variant="ghost"
          onPress={() => openSearch()}
          aria-label="Search"
          className="w-9 h-9 rounded-xl text-foreground/70 hover:text-foreground hover:bg-foreground/5 border border-transparent hover:border-border/40 transition-all cursor-pointer"
        >
          <Search size={16} />
        </Button>

        <div className="h-5 w-[1px] bg-border/50" />

        <div className="hidden sm:flex flex-col items-end">
          <p
            suppressHydrationWarning
            className="text-foreground text-xs md:text-sm font-bold truncate"
          >
            {name || "User"}
          </p>
          {persona && (
            <Chip
              size="sm"
              className="bg-primary/10 border border-primary/25 rounded-md px-1.5 py-0.5 mt-0.5"
            >
              <Chip.Label className="text-primary font-bold text-[10px] md:text-[11px] uppercase tracking-wider">
                {persona}
              </Chip.Label>
            </Chip>
          )}
        </div>

        <Dropdown>
          <Dropdown.Trigger className="outline-none rounded-full cursor-pointer focus:ring-2 focus:ring-primary/40">
            <Avatar className="w-9 h-9 bg-primary/90 text-primary-foreground font-bold text-sm flex items-center justify-center shadow-sm">
              <Avatar.Fallback>{name?.charAt(0) || "U"}</Avatar.Fallback>
            </Avatar>
          </Dropdown.Trigger>
          <Dropdown.Popover className="bg-popover/95 backdrop-blur-2xl border border-border shadow-2xl rounded-xl w-60 p-1.5">
            <Dropdown.Menu
              onAction={(key) => {
                if (key === "signout") handleLogout();
                else if (key === "connections") router.push("/connections");
                else if (key === "transactions") router.push("/transactions");
              }}
            >
              <Dropdown.Item
                id="user-info"
                textValue={name || "User"}
                className="flex flex-col gap-0.5 px-3 py-2 pointer-events-none opacity-80 border-b border-border/50 pb-2 mb-1"
              >
                <span className="text-sm font-bold text-foreground">
                  {name || "User"}
                </span>
                <span className="text-xs text-muted-foreground truncate">
                  {email || "Authenticated Session"}
                </span>
              </Dropdown.Item>
              <Dropdown.Item
                id="connections"
                textValue="Bank Connections"
                className="px-3 py-2.5 rounded-lg cursor-pointer text-sm font-medium hover:bg-foreground/5"
              >
                Bank Connections
              </Dropdown.Item>
              <Dropdown.Item
                id="transactions"
                textValue="Transactions"
                className="px-3 py-2.5 rounded-lg cursor-pointer text-sm font-medium hover:bg-foreground/5"
              >
                Transactions
              </Dropdown.Item>
              <Dropdown.Item
                id="signout"
                textValue="Sign Out"
                variant="danger"
                className="px-3 py-2.5 rounded-lg cursor-pointer text-sm font-medium text-danger hover:bg-danger/10 flex items-center gap-2"
              >
                <LogOut size={15} />
                <span>Sign Out</span>
              </Dropdown.Item>
            </Dropdown.Menu>
          </Dropdown.Popover>
        </Dropdown>
      </div>
    </header>
  );
};
