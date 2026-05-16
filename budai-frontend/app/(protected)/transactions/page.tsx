"use client";

import React, { useState } from "react";
import LedgerTableWidget from "@/app/(protected)/_components/LedgerTableWidget";
import {
  Search,
  Bell,
  LayoutDashboard,
  ArrowRightLeft,
  RefreshCcw,
  CreditCard,
  LineChart,
  BrainCircuit,
} from "lucide-react";
import { Button, Input, Link, Avatar, Tooltip } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import { useQueryClient } from "@tanstack/react-query";

export default function TransactionsPage() {
  const { userName } = useBudAI();
  const queryClient = useQueryClient();
  const [isRetraining, setIsRetraining] = useState(false);

  const handleRetrainAll = async () => {
    setIsRetraining(true);
    try {
      const res = await apiFetch("/api/categorizer/retrain", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ force: true }),
      }, true);

      if (res.ok) {
        const data = await res.json();
        const taskId = data.task_id;

        if (taskId) {
          const pollInterval = setInterval(async () => {
            try {
              const statusRes = await apiFetch(`/api/categorizer/task-status/${taskId}`, {}, true);
              if (statusRes.ok) {
                const statusData = await statusRes.json();
                if (statusData.status === "completed" || statusData.status === "failed") {
                  clearInterval(pollInterval);
                  queryClient.invalidateQueries({ queryKey: ["transactions"] });
                  setIsRetraining(false);
                }
              }
            } catch (e) {
              clearInterval(pollInterval);
              setIsRetraining(false);
            }
          }, 2000);
        } else {
          queryClient.invalidateQueries({ queryKey: ["transactions"] });
          setIsRetraining(false);
        }
      } else {
        setIsRetraining(false);
      }
    } catch (e) {
      console.error(e);
      setIsRetraining(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-obsidian font-geist overflow-hidden">

      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[10%] -left-[5%] w-[70%] h-[70%] rounded-full bg-neon-cyan/5 blur-[180px]"></div>
        <div className="absolute -bottom-[10%] -right-[5%] w-[70%] h-[70%] rounded-full bg-deep-pink/5 blur-[180px]"></div>
      </div>

      <div className="relative z-10 w-64 h-full obsidian-glass flex flex-col justify-between py-8 px-6 shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.2)]">
        <div>
          <div className="flex items-center gap-3 mb-12">
            <div className="w-8 h-8 rounded-lg bg-linear-to-br from-neon-cyan to-[#0088FF] flex items-center justify-center shadow-[0_0_15px_rgba(0,229,255,0.4)]">
              <span className="text-obsidian font-black text-lg leading-none tracking-tighter">
                B
              </span>
            </div>
            <h1 className="text-white text-2xl font-bold tracking-tight">
              BudAI
            </h1>
          </div>
          <nav className="space-y-2">
            <Link
              href="/home"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <LayoutDashboard size={20} />
              <span className="font-medium text-sm">Dashboard</span>
            </Link>
            <Link
              href="/transactions"
              className="flex items-center gap-4 text-neon-cyan bg-neon-cyan/10 px-4 py-3 rounded-xl shadow-[inset_0_0_20px_rgba(0,229,255,0.05)] transition-all border border-neon-cyan/20"
            >
              <ArrowRightLeft size={20} />
              <span className="font-semibold text-sm">Transactions</span>
            </Link>
            <Link
              href="/forecasting"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <LineChart size={20} />
              <span className="font-medium text-sm">Forecasting</span>
            </Link>
            <Link
              href="/health"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <RefreshCcw size={20} />
              <span className="font-medium text-sm">Health Radar</span>
            </Link>
            <Link
              href="/connections"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <CreditCard size={20} />
              <span className="font-medium text-sm">Connections</span>
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-3 pt-6 border-t border-white/8 mt-8">
          <Avatar className="w-10 h-10 bg-linear-to-br from-neon-cyan to-deep-pink shrink-0 shadow-[0_0_15px_rgba(255,51,102,0.3)] border border-white/10 text-white font-bold" />
          <div className="overflow-hidden">
            <p
              suppressHydrationWarning
              className="text-white text-sm font-semibold truncate"
            >
              {userName || "User"}
            </p>
            <p className="text-neon-cyan/70 font-medium text-xs truncate tracking-wide">
              BudAI Member
            </p>
          </div>
        </div>
      </div>

      <div className="relative z-10 flex-1 flex flex-col pt-8 px-8 h-full">
        <div className="flex items-center justify-between mb-8 shrink-0">
          <div className="relative w-75">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#5E6272] pointer-events-none z-10" />
            <Input
              type="text"
              placeholder="Search transactions..."
              className="obsidian-glass rounded-xl focus-within:border-neon-cyan/50 shadow-[0_4px_20px_rgba(0,0,0,0.3)] [&_input]:pl-11 [&_input]:text-sm [&_input]:text-white [&_input::placeholder]:text-[#5E6272]"
            />
          </div>
          <div className="flex items-center gap-4">
            <Tooltip content="Retrain AI Model" placement="bottom">
              <Button
                isIconOnly
                variant="secondary"
                isLoading={isRetraining}
                onPress={handleRetrainAll}
                className="w-11 h-11 min-w-11 rounded-full obsidian-glass text-neon-cyan shadow-[0_4px_20px_rgba(0,0,0,0.3)] relative cursor-pointer"
              >
                <BrainCircuit size={18} />
              </Button>
            </Tooltip>

            <Button
              isIconOnly
              variant="secondary"
              className="w-11 h-11 min-w-11 rounded-full obsidian-glass text-white shadow-[0_4px_20px_rgba(0,0,0,0.3)] relative cursor-pointer"
            >
              <Bell size={18} />
              <span className="absolute top-3 right-3 w-2.5 h-2.5 bg-deep-pink rounded-full border-2 border-obsidian shadow-[0_0_8px_rgba(255,51,102,0.8)]"></span>
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-hidden pb-8 relative">
          <div className="h-full w-full obsidian-glass rounded-3xl overflow-hidden [&::-webkit-scrollbar]:hidden">
            <LedgerTableWidget />
          </div>
        </div>
      </div>
    </div>
  );
}
