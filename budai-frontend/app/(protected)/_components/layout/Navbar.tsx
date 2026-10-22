"use client";

import React, { useState, useEffect } from "react";
import { Link, Avatar, Chip, Button, Dropdown } from "@heroui/react";
import { usePathname, useRouter } from "next/navigation";
import Image from "next/image";
import NextLink from "next/link";
import {
  LayoutDashboard,
  CreditCard,
  MessageSquare,
  Search,
  LogOut,
  Menu,
  X,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { apiFetch, clearSession } from "@/lib/api";
import { useSearch } from "@/app/context/SearchContext";

export const TopNavbar = () => {
  const pathname = usePathname();
  const router = useRouter();
  const { openSearch } = useSearch();
  const [Persona, setPersona] = useState<string | null>(null);
  const [name, setName] = useState<string | null>(null);
  const [email, setEmail] = useState<string | null>(null);
  const [isMobileMenuOpen, setIsMobileMenuOpen] = useState(false);

  useEffect(() => {
    async function fetchPersona() {
      try {
        const res = await apiFetch(`/api/auth/me`, {}, true);
        if (res.ok) {
          const json = (await res.json()) as { persona: string, name: string, email: string };
          setPersona(json.persona);
          setName(json.name);
          setEmail(json.email);
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

  const handleLogout = () => {
    clearSession();
    router.push("/login");
  };

  return (
    <header className="relative w-full h-24 bg-background/80 backdrop-blur-md border-b-[0.5px] border-border/40 flex justify-between items-center px-6 md:px-10 shrink-0 shadow-sm z-40 transition-colors">
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
        <Button
          isIconOnly
          variant="ghost"
          onPress={() => openSearch()}
          aria-label="Search"
          className="w-9 h-9 rounded-xl text-foreground/50 hover:text-foreground hover:bg-white/5 border border-transparent hover:border-white/10 transition-all cursor-pointer"
        >
          <Search size={16} />
        </Button>

        <div className="flex items-center gap-3 border-l-[0.5px] border-border/40 pl-4 ml-1">
          <div className="hidden sm:flex flex-col items-end">
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

          <Dropdown>
            <Dropdown.Trigger className="outline-none rounded-full cursor-pointer focus:ring-2 focus:ring-primary/50">
              <Avatar className="w-8 h-8 bg-primary shadow-[0_0_15px_rgba(0,242,255,0.3)] border border-white/10 text-primary-foreground font-black text-sm flex items-center justify-center">
                <Avatar.Fallback>{name?.charAt(0) || "U"}</Avatar.Fallback>
              </Avatar>
            </Dropdown.Trigger>
            <Dropdown.Popover className="bg-black/90 backdrop-blur-3xl border border-white/10 shadow-2xl rounded-2xl w-60 p-2">
              <Dropdown.Menu onAction={(key) => {
                if (key === "signout") handleLogout();
                else if (key === "connections") router.push("/connections");
                else if (key === "dashboard") router.push("/home");
              }}>
                <Dropdown.Item id="user-info" textValue={name || "User"} className="flex flex-col gap-0.5 px-3 py-2 pointer-events-none opacity-80 border-b border-white/5 pb-3 mb-1">
                  <span className="text-xs font-bold text-foreground">{name || "User"}</span>
                  <span className="text-[10px] font-mono text-muted-foreground truncate">{email || "Authenticated Session"}</span>
                </Dropdown.Item>
                <Dropdown.Item id="dashboard" textValue="Dashboard" className="px-3 py-2 rounded-xl cursor-pointer hover:bg-white/5">
                  <div className="flex items-center gap-2">
                    <LayoutDashboard size={14} className="text-primary" />
                    <span className="text-xs font-bold uppercase tracking-wider">Dashboard</span>
                  </div>
                </Dropdown.Item>
                <Dropdown.Item id="connections" textValue="Connections" className="px-3 py-2 rounded-xl cursor-pointer hover:bg-white/5">
                  <div className="flex items-center gap-2">
                    <CreditCard size={14} className="text-primary" />
                    <span className="text-xs font-bold uppercase tracking-wider">Connections</span>
                  </div>
                </Dropdown.Item>
                <Dropdown.Item id="signout" textValue="Sign Out" variant="danger" className="px-3 py-2.5 rounded-xl cursor-pointer text-danger hover:bg-danger/10">
                  <div className="flex items-center gap-2">
                    <LogOut size={14} />
                    <span className="text-xs font-bold uppercase tracking-wider">Sign Out</span>
                  </div>
                </Dropdown.Item>
              </Dropdown.Menu>
            </Dropdown.Popover>
          </Dropdown>

          <Button
            isIconOnly
            variant="ghost"
            onPress={() => setIsMobileMenuOpen(!isMobileMenuOpen)}
            aria-label="Toggle navigation menu"
            className="md:hidden w-9 h-9 rounded-xl text-foreground/60 hover:text-foreground hover:bg-white/5"
          >
            {isMobileMenuOpen ? <X size={18} /> : <Menu size={18} />}
          </Button>
        </div>
      </div>

      {isMobileMenuOpen && (
        <div className="absolute top-24 left-0 w-full bg-background/95 backdrop-blur-2xl border-b border-border/40 p-4 flex flex-col gap-2 md:hidden z-50 shadow-2xl">
          {navItems.map((item) => {
            const isActive = pathname.startsWith(item.href);
            return (
              <NextLink
                key={item.href}
                href={item.href}
                onClick={() => setIsMobileMenuOpen(false)}
                className={cn(
                  "flex items-center gap-3 px-4 py-3 rounded-xl transition-all",
                  isActive
                    ? "text-primary bg-primary/10 border border-primary/20"
                    : "text-foreground/70 hover:text-foreground hover:bg-white/5",
                )}
              >
                <item.icon size={16} />
                <span className="text-xs font-bold uppercase tracking-widest">{item.label}</span>
              </NextLink>
            );
          })}
        </div>
      )}
    </header>
  );
};
