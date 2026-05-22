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
} from "lucide-react";
import { useBudAI } from "@/app/context/AppContext";
import { cn } from "@/lib/utils";

export const Sidebar = () => {
  const pathname = usePathname();
  const { userName } = useBudAI();

  const navItems = [
    {
      href: "/home",
      label: "Dashboard",
      icon: LayoutDashboard,
    },
    {
      href: "/transactions",
      label: "Transactions",
      icon: ArrowRightLeft,
    },
    {
      href: "/forecasting",
      label: "Projections",
      icon: LineChart,
    },
    {
      href: "/health",
      label: "Financial Health",
      icon: RefreshCcw,
    },
    {
      href: "/connections",
      label: "Your Connections",
      icon: CreditCard,
    },
  ];

  return (
    <div className="relative z-10 w-60 h-full bg-black/40 backdrop-blur-3xl border-r-[0.5px] border-white/10 flex flex-col justify-between py-10 px-6 shrink-0 shadow-inner">
      <div>
        <div className="flex items-center gap-3 mb-16 px-2">
          <div className="w-7 h-7 rounded-md bg-primary flex items-center justify-center shadow-[0_0_15px_rgba(0,127,255,0.4)]">
            <span className="text-primary-foreground font-black text-sm leading-none tracking-tighter m-0 p-0">
              B
            </span>
          </div>
          <h1 className="text-foreground text-xl font-bold tracking-tighter uppercase italic">
            BudAI
          </h1>
        </div>
        <nav className="space-y-1.5">
          {navItems.map((item) => {
            const isActive = pathname === item.href;
            return (
              <Link
                key={item.href}
                href={item.href}
                className={cn(
                  "flex items-center gap-4 px-4 py-3 rounded-xl transition-all border-[0.5px]",
                  isActive
                    ? "text-primary bg-primary/10 backdrop-blur-xl shadow-[0_0_20px_rgba(0,127,255,0.05)] border-primary/30"
                    : "text-foreground/40 hover:text-foreground hover:bg-white/5 border-transparent hover:border-white/5"
                )}
              >
                <item.icon size={18} />
                <span className="text-[10px] font-black uppercase tracking-[0.2em]">
                  {item.label}
                </span>
              </Link>
            );
          })}
        </nav>
      </div>

      <div className="flex items-center gap-3 pt-6 border-t-[0.5px] border-white/5 mt-8 px-2">
        <Avatar className="w-8 h-8 bg-primary shrink-0 shadow-[0_0_15px_rgba(0,127,255,0.3)] border-[0.5px] border-white/10 text-primary-foreground font-black text-[10px]">
           <Avatar.Fallback>{userName?.charAt(0) || "U"}</Avatar.Fallback>
        </Avatar>
        <div className="overflow-hidden">
          <p
            suppressHydrationWarning
            className="text-foreground text-[11px] font-bold truncate uppercase tracking-tighter"
          >
            {userName || "User"}
          </p>
          <p className="text-primary/50 font-black text-[8px] truncate tracking-[0.2em] uppercase">
            Core Node
          </p>
        </div>
      </div>
    </div>
  );
};
