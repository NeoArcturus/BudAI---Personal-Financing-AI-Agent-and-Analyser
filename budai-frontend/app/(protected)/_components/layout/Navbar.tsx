"use client";

import React, { useState, useEffect } from "react";
import { Link, Avatar, Chip } from "@heroui/react";
import { usePathname } from "next/navigation";
import Image from "next/image";
import NextLink from "next/link";
import {
  LayoutDashboard,
  CreditCard,
  MessageSquare,
  Search,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { apiFetch, getUserUuid } from "@/lib/api";

export const TopNavbar = () => {
  const pathname = usePathname();
  const [Persona, setPersona] = useState<string | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [email, setEmail] = useState<string | null>(null);

  useEffect(() => {
    async function fetchPersona() {
      try {
        const res = await apiFetch(`/api/auth/me`, {}, true);
        if (res.ok) {
          const json = (await res.json()) as { persona: string, name: string, email: string }
          setPersona(json.persona)
          setName(json.name)
          setEmail(json.email)
        }
      } catch (err) {
        console.error("Failed to fetch macro persona", err);
      }
    }
    fetchPersona();
  }, []);

  const navItems = [
    { href: "/home", label: "Dashboard", icon: LayoutDashboard },
    { href: "/advisor", label: "Advisor", icon: MessageSquare },
    { href: "/connections", label: "Connections", icon: CreditCard },
  ];

  return (
    <header className="relative z-50 w-full h-24 bg-[#0c131d] border-b-[0.5px] border-white/5 flex justify-between items-center px-6 md:px-10 shrink-0 shadow-sm">
      <div className="flex items-center justify-start shrink-0">
        <NextLink href="/" className="flex items-center">
          <Image
            src="/FullLogo.jpg"
            alt="BudAI Logo"
            width={80}
            height={25}
            className="rounded-sm object-contain"
            priority
          />
        </NextLink>
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
                  ? "text-primary bg-primary/10 border-primary/20 shadow-[0_0_15px_rgba(0,242,255,0.05)]"
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
        <div className="hidden md:flex items-center gap-3 border-l-[0.5px] border-white/10 pl-4 ml-1">
          <div className="flex flex-col items-end">
            <p
              suppressHydrationWarning
              className="text-foreground text-[11px] font-bold truncate uppercase tracking-tighter"
            >
              {name || "User"}
            </p>
            {Persona && (
              <Chip
                size="sm"
                className="bg-primary/10 border-[0.5px] border-primary/30 mt-0.5 rounded-lg items-center justify-center flex flex-row"
              >
                <Chip.Label className="text-primary font-black text-[8px] tracking-[0.2em] uppercase px-1 py-0.5">
                  {Persona}
                </Chip.Label>
              </Chip>
            )}
          </div>
          <Avatar className="w-8 h-8 bg-primary shadow-[0_0_15px_rgba(0,242,255,0.3)] border border-white/10 text-primary-foreground font-black text-sm flex items-center justify-center cursor-pointer">
            <Avatar.Fallback>{name?.charAt(0) || "U"}</Avatar.Fallback>
          </Avatar>
        </div>
      </div>
    </header>
  );
};
