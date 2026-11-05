"use client";

import React from "react";
import NextLink from "next/link";
import { usePathname } from "next/navigation";
import {
  LayoutDashboard,
  ReceiptText,
  MessageSquare,
  Landmark,
} from "lucide-react";
import { cn } from "@/lib/utils";

interface NavItem {
  href: string;
  label: string;
  icon: React.ComponentType<{ size?: number; className?: string }>;
}

const navItems: NavItem[] = [
  { href: "/home", label: "Dashboard", icon: LayoutDashboard },
  { href: "/transactions", label: "Transactions", icon: ReceiptText },
  { href: "/advisor", label: "Advisor", icon: MessageSquare },
  { href: "/connections", label: "Connections", icon: Landmark },
];

export const FloatingBottomNav = () => {
  const pathname = usePathname();

  return (
    <nav
      aria-label="Main Navigation"
      className="fixed bottom-7 left-1/2 -translate-x-1/2 z-50 select-none print:hidden pointer-events-auto"
    >
      <div className="flex items-center gap-2 px-4 py-2.5 rounded-full bg-neutral-900/95 border border-white/15 shadow-[0_15px_45px_rgba(0,0,0,0.45)] transition-all">
        {navItems.map((item) => {
          const isActive =
            item.href === "/home"
              ? pathname === "/home"
              : pathname.startsWith(item.href);

          const Icon = item.icon;

          return (
            <NextLink
              key={item.href}
              href={item.href}
              className={cn(
                "relative flex items-center gap-2.5 px-5 py-2.5 rounded-full transition-all duration-200 text-xs md:text-sm font-bold tracking-wider",
                isActive
                  ? "bg-primary text-primary-foreground shadow-lg shadow-primary/30"
                  : "text-foreground/75 hover:text-foreground hover:bg-foreground/10",
              )}
            >
              <Icon
                size={18}
                className={cn(
                  isActive ? "text-primary-foreground" : "text-foreground/70",
                )}
              />
              <span className="uppercase">{item.label}</span>
            </NextLink>
          );
        })}
      </div>
    </nav>
  );
};
