"use client";

import React from "react";
import { Link, Avatar } from "@heroui/react";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ArrowRightLeft,
  LineChart,
  RefreshCcw,
  CreditCard,
  MessageSquare,
} from "lucide-react";
import { useBudAI } from "@/app/context/AppContext";
import { cn } from "@/lib/utils";

export const TopNavbar = () => {
  const pathname = usePathname();
  const { userName } = useBudAI();

  const navItems = [
    { href: "/home", label: "Dashboard", icon: LayoutDashboard },
    { href: "/transactions", label: "Transactions", icon: ArrowRightLeft },
    { href: "/forecasting", label: "Projections", icon: LineChart },
    { href: "/health", label: "Health", icon: RefreshCcw },
    { href: "/connections", label: "Connections", icon: CreditCard },
    { href: "/advisor", label: "Advisor", icon: MessageSquare },
  ];

  return (
    <header className="relative z-50 w-full h-16 bg-background/80 backdrop-blur-3xl border-b border-white/10 grid grid-cols-3 items-center px-6 shrink-0 shadow-sm">
      <div className="flex items-center justify-start gap-3">
        <div className="w-7 h-7 rounded-md bg-primary flex items-center justify-center shadow-[0_0_15px_rgba(0,127,255,0.4)]">
          <span className="text-primary-foreground font-black text-sm leading-none tracking-tighter m-0 p-0">
            B
          </span>
        </div>
        <h1 className="text-foreground text-xl font-bold tracking-tighter uppercase italic">
          BudAI
        </h1>
      </div>
        
      <nav className="hidden md:flex items-center justify-center gap-2">
        {navItems.map((item) => {
          const isActive = pathname.startsWith(item.href);
          return (
            <Link
              key={item.href}
              href={item.href}
              className={cn(
                "flex items-center gap-2 px-3 py-2 rounded-lg transition-all border border-transparent",
                isActive
                  ? "text-primary bg-primary/10 border-primary/20 shadow-[0_0_15px_rgba(0,127,255,0.05)]"
                  : "text-foreground/50 hover:text-foreground hover:bg-white/5 hover:border-white/10",
              )}
            >
              <item.icon size={14} />
              <span className="text-[10px] font-black uppercase tracking-[0.1em]">
                {item.label}
              </span>
            </Link>
          );
        })}
      </nav>

      <div className="flex items-center justify-end gap-3 pl-6">
        <div className="hidden md:flex flex-col items-end">
          <p
            suppressHydrationWarning
            className="text-foreground text-[11px] font-bold truncate uppercase tracking-tighter"
          >
            {userName || "User"}
          </p>
          <p className="text-primary/50 font-black text-[8px] tracking-[0.2em] uppercase">
            Primary Profile
          </p>
        </div>
        <Avatar className="w-8 h-8 bg-primary shadow-[0_0_15px_rgba(0,127,255,0.3)] border border-white/10 text-primary-foreground font-black text-sm flex items-center justify-center cursor-pointer">
          <Avatar.Fallback>{userName?.charAt(0) || "U"}</Avatar.Fallback>
        </Avatar>
      </div>
    </header>
  );
};
