"use client";

import React, { useState, useEffect } from "react";
import { Modal, Input, ListBox } from "@heroui/react";
import { Search, MessageSquare, Compass, Building2, ReceiptText, Calendar, LucideIcon } from "lucide-react";
import { useSearch } from "@/app/context/SearchContext";
import { useBudAI } from "@/app/context/AppContext";
import { useRouter } from "next/navigation";
import { useSearchTransactions } from "@/lib/hooks";
import { Transaction } from "@/types";

interface SearchOption {
  id: string;
  type: "action" | "nav" | "account" | "transaction";
  label: string;
  icon: LucideIcon;
  section: string;
  path?: string;
  amount?: number;
  isPositive?: boolean;
  date?: string;
  category?: string;
  txData?: Transaction;
}

function useDebounce<T>(value: T, delay: number): T {
  const [debouncedValue, setDebouncedValue] = useState<T>(value);
  useEffect(() => {
    const handler = setTimeout(() => {
      setDebouncedValue(value);
    }, delay);
    return () => clearTimeout(handler);
  }, [value, delay]);
  return debouncedValue;
}

export default function GlobalSearchModal() {
  const { isOpen, closeSearch } = useSearch();
  const { accounts, sendChatMessage } = useBudAI();
  const router = useRouter();
  const [query, setQuery] = useState("");
  const debouncedQuery = useDebounce(query, 300);

  const { data: txResults } = useSearchTransactions(debouncedQuery);

  
  useEffect(() => {
    if (isOpen) {
      setQuery("");
    }
  }, [isOpen]);

  
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === "Escape" && isOpen) {
        e.preventDefault();
        closeSearch();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [isOpen, closeSearch]);

  if (!isOpen) return null;

  const staticOptions: SearchOption[] = [
    {
      id: "ai-prompt",
      type: "action",
      label: query ? `Ask AI: "${query}"` : "Ask AI...",
      icon: MessageSquare,
      section: "AI Advisor",
    },
    {
      id: "nav-home",
      type: "nav",
      label: "Go to Dashboard",
      icon: Compass,
      section: "Navigation",
      path: "/home",
    },
    {
      id: "nav-settings",
      type: "nav",
      label: "Go to Settings",
      icon: Compass,
      section: "Navigation",
      path: "/settings",
    },
    ...accounts.map((acc) => ({
      id: `acc-${acc.account_id}`,
      type: "account" as const,
      label: acc.bank_name || acc.provider_name || "Account",
      icon: Building2,
      section: "Accounts",
      path: `/accounts/${acc.account_id}`,
    })),
  ];

  const transactionOptions: SearchOption[] = (txResults || []).map((tx) => {
    const isPositive = (tx.amount || 0) >= 0;
    return {
      id: `tx-${tx.transaction_id || tx.transaction_uuid || Math.random()}`,
      type: "transaction",
      label: tx.description || tx.merchant_name || "Unknown Transaction",
      icon: ReceiptText,
      section: "Transactions",
      amount: tx.amount,
      isPositive,
      date: tx.date || tx.timestamp ? new Date(tx.date || tx.timestamp || "").toLocaleDateString() : "",
      category: tx.category || tx.Category || "",
      txData: tx,
    };
  });

  const filteredStatic = query
    ? staticOptions.filter(
        (o) =>
          o.type === "action" ||
          (o.label && o.label.toLowerCase().includes(query.toLowerCase())),
      )
    : staticOptions;

  const filteredOptions: SearchOption[] = [...filteredStatic, ...transactionOptions];

  const handleAction = (key: React.Key) => {
    const item = filteredOptions.find((o) => o.id === key);
    if (!item) return;

    closeSearch();

    if (item.type === "action") {
      if (query.trim()) {
        sendChatMessage(query);
      }
      router.push("/home");
    } else if ((item.type === "nav" || item.type === "account") && item.path) {
      router.push(item.path);
    } else if (item.type === "transaction") {
      router.push("/transactions");
    }
  };

  return (
    <Modal
      isOpen={isOpen}
      onOpenChange={(open) => !open && closeSearch()}
    >
      <Modal.Backdrop className="bg-black/60 backdrop-blur-md fixed inset-0 z-[200]" />
      <Modal.Container className="fixed top-[15vh] left-1/2 -translate-x-1/2 z-[201] w-full max-w-2xl px-4 pointer-events-auto">
        <Modal.Dialog className="bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-2xl shadow-2xl overflow-hidden">
          <div className="p-4 border-b-[0.5px] border-white/5 flex items-center">
            <Search className="text-primary/50 mx-2" size={20} />
            <Input
              autoFocus
              className="w-full text-foreground text-lg font-bold placeholder:text-foreground/30 bg-transparent border-none shadow-none focus:outline-none"
              placeholder="Search accounts, navigation, or ask AI..."
              value={query}
              onChange={(e) => setQuery(e.target.value)}
            />
          </div>
          <div className="p-2 max-h-[60vh] overflow-y-auto scrollbar-hide">
            {filteredOptions.length === 0 ? (
              <div className="p-4 text-center text-foreground/30 text-xs font-bold uppercase tracking-widest">
                No results found.
              </div>
            ) : (
              <ListBox
                items={filteredOptions}
                onAction={handleAction}
                className="p-0"
              >
                {(item) => {
                  const Icon = item.icon;
                  return (
                    <ListBox.Item
                      key={item.id}
                      id={item.id}
                      textValue={item.label}
                      className="p-2 rounded-xl hover:bg-white/5 cursor-pointer flex items-center gap-4 transition-all group data-[hover=true]:bg-white/5 outline-none data-[focus-visible=true]:ring-2 data-[focus-visible=true]:ring-primary/50"
                    >
                      <div className="flex items-center gap-4 w-full">
                        <div className="w-10 h-10 shrink-0 rounded-lg bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shadow-[0_0_15px_rgba(0,242,255,0.05)]">
                          <Icon size={18} />
                        </div>
                        <div className="flex flex-col overflow-hidden w-full">
                          <span className="text-primary/50 text-[9px] font-black uppercase tracking-widest mb-0.5">
                            {item.section}
                          </span>
                          <div className="flex justify-between items-center w-full pr-2">
                            <span className="text-foreground text-[13px] font-bold truncate">
                              {item.label}
                            </span>
                            {item.type === "transaction" && (
                              <div className="flex items-center gap-4 shrink-0 ml-4">
                                <div className="flex items-center gap-1.5 text-foreground/40 hidden sm:flex">
                                  <Calendar size={12} />
                                  <span className="text-[10px] font-medium tracking-wide">{item.date}</span>
                                </div>
                                {item.category && (
                                  <span className="text-[9px] px-2 py-0.5 rounded-full bg-white/5 border-[0.5px] border-white/10 text-foreground/60 font-black uppercase tracking-widest hidden md:inline-block">
                                    {item.category}
                                  </span>
                                )}
                                <span className={`text-[12px] font-black uppercase tracking-tight ${item.isPositive ? "text-green-500" : "text-foreground"}`}>
                                  {item.isPositive ? "+" : ""}£{Math.abs(item.amount || 0).toFixed(2)}
                                </span>
                              </div>
                            )}
                          </div>
                        </div>
                      </div>
                    </ListBox.Item>
                  );
                }}
              </ListBox>
            )}
          </div>
          <div className="p-3 border-t-[0.5px] border-white/5 bg-white/5 flex items-center justify-between">
            <span className="text-foreground/30 text-[9px] font-black uppercase tracking-widest">
              Pro-Tip
            </span>
            <span className="text-foreground/40 text-[9px] font-bold uppercase tracking-widest flex items-center gap-2">
              Use <kbd className="px-1.5 py-0.5 bg-black/50 border-[0.5px] border-white/10 rounded font-mono text-[9px]">↑</kbd> <kbd className="px-1.5 py-0.5 bg-black/50 border-[0.5px] border-white/10 rounded font-mono text-[9px]">↓</kbd> to navigate
              <span className="mx-1">•</span>
              <kbd className="px-1.5 py-0.5 bg-black/50 border-[0.5px] border-white/10 rounded font-mono text-[9px]">↵</kbd> to select
            </span>
          </div>
        </Modal.Dialog>
      </Modal.Container>
    </Modal>
  );
}
