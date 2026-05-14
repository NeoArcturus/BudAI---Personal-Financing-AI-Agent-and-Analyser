"use client";

import React, { useState, useEffect } from "react";
import {
  LayoutDashboard,
  ArrowRightLeft,
  RefreshCcw,
  CreditCard,
  ShieldCheck,
  ExternalLink,
  AlertTriangle,
  LineChart,
  Plus,
  CheckCircle2,
  Trash2,
} from "lucide-react";
import { Button, Link, Avatar, Card, Chip } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";

export default function ConnectionsPage() {
  const { userName, accounts } = useBudAI();
  const [, setConnectionStatus] = useState(null);
  const [isConnecting, setIsConnecting] = useState(false);

  useEffect(() => {
    apiFetch("/api/auth/truelayer/status", {}, true)
      .then((res) => res.json())
      .then((data) => setConnectionStatus(data))
      .catch((err) => console.error(err));
  }, []);

  const handleConnect = async () => {
    setIsConnecting(true);
    try {
      const res = await apiFetch("/api/auth/truelayer/status", {}, true);
      if (res.ok) {
        const data = await res.json();
        if (data.auth_url) {
          window.location.href = data.auth_url;
        }
      }
    } catch (error) {
      console.error(error);
    } finally {
      setIsConnecting(false);
    }
  };

  return (
    <div className="flex h-screen w-full bg-obsidian font-geist overflow-hidden">
      {/* Background Glows (Slightly increased opacity to 10 for depth) */}
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[10%] -left-[5%] w-[70%] h-[70%] rounded-full bg-neon-cyan/10 blur-[180px]"></div>
        <div className="absolute -bottom-[10%] -right-[5%] w-[70%] h-[70%] rounded-full bg-deep-pink/10 blur-[180px]"></div>
      </div>

      {/* Sidebar */}
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
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all border border-transparent"
            >
              <LayoutDashboard size={20} />
              <span className="font-medium text-sm">Dashboard</span>
            </Link>
            <Link
              href="/transactions"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all border border-transparent"
            >
              <ArrowRightLeft size={20} />
              <span className="font-medium text-sm">Transactions</span>
            </Link>
            <Link
              href="/forecasting"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all border border-transparent"
            >
              <LineChart size={20} />
              <span className="font-medium text-sm">Forecasting</span>
            </Link>
            <Link
              href="/health"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all border border-transparent"
            >
              <RefreshCcw size={20} />
              <span className="font-medium text-sm">Health Radar</span>
            </Link>
            <Link
              href="/connections"
              className="flex items-center gap-4 text-neon-cyan bg-neon-cyan/10 px-4 py-3 rounded-xl shadow-[inset_0_0_20px_rgba(0,229,255,0.05)] transition-all border border-neon-cyan/20"
            >
              <CreditCard size={20} />
              <span className="font-semibold text-sm">Connections</span>
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-3 pt-6 border-t border-white/10 mt-8">
          <Avatar className="w-10 h-10 bg-linear-to-br from-neon-cyan to-deep-pink shrink-0 shadow-[0_0_15px_rgba(255,51,102,0.3)] border border-white/10 text-white font-bold"></Avatar>
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

      {/* Main Content */}
      <div className="relative z-10 flex-1 flex flex-col pt-8 px-8 h-full">
        <div className="flex items-center justify-between mb-8 shrink-0">
          <div>
            <h2 className="text-white text-3xl font-bold tracking-tight">
              Bank Connections
            </h2>
            <p className="text-white/70 font-medium tracking-wide text-sm mt-1">
              Manage your financial data sources and permissions
            </p>
          </div>
          <Button
            onPress={handleConnect}
            isPending={isConnecting}
            className="flex items-center justify-center gap-2 bg-neon-cyan text-obsidian font-bold rounded-xl px-6 h-12 shadow-[0_0_20px_rgba(0,229,255,0.3)] hover:shadow-[0_0_30px_rgba(0,229,255,0.6)] hover:scale-105 transition-all duration-300 border-none cursor-pointer"
          >
            <Plus size={18} /> Connect New Account
          </Button>
        </div>

        <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-6 pb-8 overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
          <div className="space-y-6">
            <h3 className="text-white font-bold text-lg px-2 tracking-tight">
              Active Institutions
            </h3>
            {accounts.length === 0 && !isConnecting && (
              <Card className="obsidian-glass rounded-3xl p-12 shadow-2xl flex flex-col items-center justify-center text-center">
                <CreditCard className="w-16 h-16 text-[#5E6272] mb-4 opacity-20" />
                <p className="text-white font-bold text-xl mb-2">
                  No Connections Found
                </p>
                <p className="text-[#8B8E98] text-sm mb-8 max-w-xs font-medium">
                  Securely link your bank account to start analyzing your
                  finances with AI.
                </p>
                <Button
                  onPress={handleConnect}
                  className="bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-xl px-8 h-12 font-bold transition-all"
                >
                  Link First Account
                </Button>
              </Card>
            )}
            {accounts.map((acc, i) => (
              <Card
                key={i}
                className="obsidian-glass rounded-3xl p-6 shadow-2xl hover:border-white/20 transition-all group"
              >
                <div className="flex items-center justify-between mb-6">
                  <div className="flex items-center gap-3">
                    <div className="w-12 h-12 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center overflow-hidden shrink-0">
                      <span className="text-white font-black text-xl">
                        {acc.bank_name?.charAt(0) || "B"}
                      </span>
                    </div>
                    <div>
                      <h4 className="text-white font-bold text-lg tracking-tight">
                        {acc.bank_name}
                      </h4>
                      <p className="text-[#8B8E98] text-xs font-mono tracking-wider">
                        •••• {acc.account_number?.slice(-4) || "0000"}
                      </p>
                    </div>
                  </div>
                  <Chip className="bg-brand-green/10 text-brand-green border border-brand-green/20 px-3 py-1 font-bold uppercase tracking-wider flex items-center justify-center h-7 text-[10px]">
                    Connected
                  </Chip>
                </div>

                <div className="grid grid-cols-2 gap-4 mb-6">
                  <div className="p-3.5 rounded-2xl bg-white/5 border border-white/5">
                    <p className="text-[10px] text-[#5E6272] uppercase font-bold tracking-widest mb-1.5">
                      Last Sync
                    </p>
                    <p className="text-white text-sm font-medium">
                      2 minutes ago
                    </p>
                  </div>
                  <div className="p-3.5 rounded-2xl bg-white/5 border border-white/5">
                    <p className="text-[10px] text-[#5E6272] uppercase font-bold tracking-widest mb-1.5">
                      Expiry
                    </p>
                    <p className="text-white text-sm font-medium">
                      84 days left
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-3">
                  <Button
                    onPress={handleConnect}
                    className="flex-1 flex items-center justify-center bg-white/5 hover:bg-white/10 text-white border border-white/10 rounded-xl h-11 font-bold transition-colors"
                  >
                    <RefreshCcw size={16} className="mr-2" /> Re-authenticate
                  </Button>
                  <Button
                    isIconOnly
                    className="w-11 min-w-11 h-11 bg-deep-pink/10 hover:bg-deep-pink/20 text-deep-pink border border-deep-pink/20 rounded-xl flex items-center justify-center transition-colors cursor-pointer"
                  >
                    <Trash2 size={18} />
                  </Button>
                </div>
              </Card>
            ))}
          </div>

          <div className="space-y-6">
            <h3 className="text-white font-bold text-lg px-2 tracking-tight">
              Security & Permissions
            </h3>
            <Card className="obsidian-glass rounded-3xl p-6 shadow-2xl">
              <div className="flex items-center gap-4 mb-6">
                <div className="w-10 h-10 rounded-xl bg-neon-cyan/10 flex items-center justify-center text-neon-cyan shrink-0">
                  <ShieldCheck size={20} />
                </div>
                <div>
                  <h4 className="text-white font-bold tracking-tight">
                    Data Sovereignty
                  </h4>
                  <p className="text-[#8B8E98] text-xs font-medium">
                    Bank-grade encryption & PSD2 compliant
                  </p>
                </div>
              </div>

              <div className="space-y-3">
                {[
                  { label: "Transaction History" },
                  { label: "Account Balances" },
                  { label: "Direct Debits" },
                  { label: "Identity Verification" },
                ].map((item, i) => (
                  <div
                    key={i}
                    className="flex items-center justify-between p-4 rounded-2xl bg-white/5 border border-white/5"
                  >
                    <span className="text-sm text-white font-medium">
                      {item.label}
                    </span>
                    <CheckCircle2
                      size={16}
                      className="text-brand-green filter-[drop-shadow(0_0_8px_rgba(0,224,150,1))_drop-shadow(0_0_24px_rgba(0,224,150,0.6))]"
                    />
                  </div>
                ))}
              </div>

              <div className="mt-8 p-5 rounded-2xl bg-deep-pink/5 border border-deep-pink/20 flex items-center gap-4">
                <AlertTriangle size={20} className="text-deep-pink shrink-0" />
                <div>
                  <p className="text-sm text-white font-bold mb-1 tracking-tight">
                    Re-authentication Required
                  </p>
                  <p className="text-xs text-[#8B8E98] leading-relaxed font-medium">
                    Some institutions require re-auth every 90 days. We&apos;ll
                    notify you 7 days before expiry.
                  </p>
                </div>
              </div>
            </Card>

            <Button className="w-full h-14 flex items-center justify-between px-6 obsidian-glass text-[#8B8E98] hover:text-white rounded-2xl cursor-pointer transition-colors">
              <span className="font-medium tracking-wide">
                View Connection Logs
              </span>
              <ExternalLink size={18} />
            </Button>
          </div>
        </div>
      </div>
    </div>
  );
}
