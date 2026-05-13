"use client";

import React from "react";
import LedgerTableWidget from "@/app/(protected)/_components/LedgerTableWidget";
import { Search, Bell, Settings, LayoutDashboard, ArrowRightLeft, RefreshCcw, CreditCard, Moon, Sun, LineChart } from "lucide-react";
import { Button, Input, Link, Avatar } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";

export default function TransactionsPage() {
  const { userName } = useBudAI();

  return (
    <div className="flex h-screen w-full bg-obsidian font-geist overflow-hidden">
      {/* Background Glows */}
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[10%] -left-[5%] w-[70%] h-[70%] rounded-full bg-neon-cyan/5 blur-[180px]"></div>
        <div className="absolute -bottom-[10%] -right-[5%] w-[70%] h-[70%] rounded-full bg-deep-pink/5 blur-[180px]"></div>
      </div>

      {/* Sidebar */}
      <div className="relative z-10 w-64 h-full bg-obsidian/40 backdrop-blur-[24px] border-r border-white/8 flex flex-col justify-between py-8 px-6 shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.2)]">
        <div>
          <div className="flex items-center gap-3 mb-12">
            <div className="w-8 h-8 rounded-lg bg-linear-to-br from-neon-cyan to-[#0088FF] flex items-center justify-center shadow-[0_0_15px_rgba(0,229,255,0.4)]">
              <span className="text-obsidian font-black text-lg leading-none tracking-tighter">B</span>
            </div>
            <h1 className="text-white text-2xl font-bold tracking-tight">BudAI</h1>
          </div>
          <nav className="space-y-2">
            <Link href="/home" className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all">
              <LayoutDashboard size={20} />
              <span className="font-medium text-sm">Dashboard</span>
            </Link>
            <Link href="/transactions" className="flex items-center gap-4 text-neon-cyan bg-neon-cyan/10 px-4 py-3 rounded-xl shadow-[inset_0_0_20px_rgba(0,229,255,0.05)] transition-all border border-neon-cyan/20">
              <ArrowRightLeft size={20} />
              <span className="font-semibold text-sm">Transactions</span>
            </Link>
            <Link href="/forecasting" className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all">
              <LineChart size={20} />
              <span className="font-medium text-sm">Forecasting</span>
            </Link>
            <Link href="/health" className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all">
              <RefreshCcw size={20} />
              <span className="font-medium text-sm">Health Radar</span>
            </Link>
            <Link href="/connections" className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all">
              <CreditCard size={20} />
              <span className="font-medium text-sm">Connections</span>
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-3 pt-6 border-t border-white/8 mt-8">
          <Avatar className="w-10 h-10 bg-linear-to-br from-neon-cyan to-deep-pink shrink-0 shadow-[0_0_15px_rgba(255,51,102,0.3)] border border-white/10 text-white font-bold" />
          <div className="overflow-hidden">
            <p suppressHydrationWarning className="text-white text-sm font-semibold truncate">{userName || "User"}</p>
            <p className="text-neon-cyan/70 font-medium text-xs truncate tracking-wide">BudAI Member</p>
          </div>
        </div>
      </div>

      {/* Main Content */}
      <div className="relative z-10 flex-1 flex flex-col pt-8 px-8 h-full">
        <div className="flex items-center justify-between mb-8 shrink-0">
          <div className="relative w-75">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#5E6272] pointer-events-none z-10" />
            <Input
              type="text"
              placeholder="Search transactions..."
              className="bg-obsidian/40 backdrop-blur-[24px] border border-white/8 rounded-xl focus-within:border-neon-cyan/50 shadow-[0_4px_20px_rgba(0,0,0,0.3)] [&_input]:pl-11 [&_input]:text-sm [&_input]:text-white [&_input::placeholder]:text-[#5E6272]"
            />
          </div>
          <div className="flex items-center gap-4">
             <Button
              isIconOnly
              variant="flat"
              className="w-11 h-11 min-w-11 rounded-full bg-obsidian/40 backdrop-blur-[24px] border border-white/8 text-white shadow-[0_4px_20px_rgba(0,0,0,0.3)] relative cursor-pointer"
            >
              <Bell size={18} />
              <span className="absolute top-3 right-3 w-2.5 h-2.5 bg-deep-pink rounded-full border-2 border-obsidian shadow-[0_0_8px_rgba(255,51,102,0.8)]"></span>
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-hidden pb-8 relative">
          <div className="h-full w-full bg-obsidian/40 backdrop-blur-[24px] rounded-3xl border border-white/8 overflow-hidden [&::-webkit-scrollbar]:hidden">
            <LedgerTableWidget />
          </div>
        </div>
      </div>
    </div>
  );
}
