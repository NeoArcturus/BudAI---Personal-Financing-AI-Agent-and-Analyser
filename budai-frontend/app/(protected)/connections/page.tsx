"use client";

import React, { useState, useEffect } from "react";
import {
  RefreshCcw,
  CreditCard,
  ShieldCheck,
  AlertTriangle,
  Plus,
  CheckCircle2,
  Trash2,
} from "lucide-react";
import { Button, Card, Skeleton } from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import { useRouter } from "next/navigation";

export default function ConnectionsPage() {
  const router = useRouter();
  const { accounts } = useBudAI();
  const [, setConnectionStatus] = useState(null);
  const [isConnecting, setIsConnecting] = useState(false);
  const [isLoading, setIsLoading] = useState(true);

  useEffect(() => {
    const timer = setTimeout(() => {
      setIsLoading(false);
    }, 1000);

    apiFetch("/api/auth/truelayer/status", {}, true)
      .then((res) => res.json() as any)
      .then((data) => {
        setConnectionStatus(data);
        if (accounts.length > 0) {
          setIsLoading(false);
          clearTimeout(timer);
        }
      })
      .catch((err) => {
        console.error(err);
        setIsLoading(false);
      });

    return () => clearTimeout(timer);
  }, [accounts]);

  const handleConnect = async () => {
    setIsConnecting(true);
    try {
      const res = await apiFetch("/api/auth/truelayer/status", {}, true);
      if (res.ok) {
        const data = await res.json() as any;
        if (data.auth_url) {
          router.push(data.auth_url);
        }
      }
    } catch (error) {
      console.error(error);
    } finally {
      setIsConnecting(false);
    }
  };

  return (
    <div className="relative z-10 flex-1 flex flex-col pt-10 px-10 h-full">
      <div className="flex items-center justify-between mb-10 shrink-0">
        <div>
          <h2 className="text-foreground text-3xl font-black tracking-tighter uppercase italic">
            Bank <span className="font-normal not-italic">Connections</span>
          </h2>
          <p className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.4em] mt-1.5">
            Secure Data Access
          </p>
        </div>
        <Button
          onPress={handleConnect}
          isPending={isConnecting}
          className="flex items-center justify-center gap-3 bg-linear-to-r from-[#7000ff] to-[#00f2ff] text-white border-none font-black text-[10px] uppercase tracking-widest rounded-xl px-8 h-12 shadow-[0_0_20px_rgba(0,242,255,0.2)] hover:shadow-[0_0_30px_rgba(0,242,255,0.4)] hover:scale-[1.02] transition-all border-none cursor-pointer"
        >
          <Plus size={16} /> Connect Bank
        </Button>
      </div>

      <div className="flex-1 grid grid-cols-1 lg:grid-cols-2 gap-8 pb-8 overflow-y-auto scrollbar-hide">
        <div className="space-y-8">
          <h3 className="text-[10px] font-black text-foreground/30 uppercase tracking-[0.4em] px-2 italic">
            Active Institutions
          </h3>
          {isLoading ? (
            Array.from({ length: 2 }).map((_, i) => (
              <Card
                key={i}
                className="liquid-glass rounded-xl p-10 border-none shadow-inner"
              >
                <div className="flex items-center justify-between mb-10">
                  <div className="flex items-center gap-5 w-full">
                    <Skeleton animationType="shimmer" className="w-14 h-14 rounded-xl bg-white/5" />
                    <div className="space-y-3">
                      <Skeleton animationType="shimmer" className="h-5 w-40 rounded bg-white/5" />
                      <Skeleton animationType="shimmer" className="h-3 w-28 rounded bg-white/5" />
                    </div>
                  </div>
                </div>
                <div className="grid grid-cols-2 gap-6 mb-10">
                  <Skeleton animationType="shimmer" className="h-16 rounded-xl bg-white/5" />
                  <Skeleton animationType="shimmer" className="h-16 rounded-xl bg-white/5" />
                </div>
                <div className="flex gap-4">
                  <Skeleton animationType="shimmer" className="h-12 flex-1 rounded-lg bg-white/5" />
                  <Skeleton animationType="shimmer" className="h-12 w-12 rounded-lg bg-white/5" />
                </div>
              </Card>
            ))
          ) : accounts.length === 0 && !isConnecting ? (
            <Card className="liquid-glass rounded-xl p-16 shadow-inner border-none flex flex-col items-center justify-center text-center">
              <div className="w-16 h-16 rounded-2xl bg-white/2 border-[0.5px] border-white/5 flex items-center justify-center text-foreground/10 mb-8">
                <CreditCard className="w-8 h-8" />
              </div>
              <p className="text-foreground font-black text-xl mb-4 tracking-[0.2em] uppercase italic">
                No Accounts linked
              </p>
              <p className="text-foreground/20 text-[10px] mb-10 max-w-xs font-black uppercase tracking-widest leading-relaxed">
                Securely authorize a bank connection to begin analyzing your finances.
              </p>
              <Button
                onPress={handleConnect}
                className="bg-white/5 border-[0.5px] border-white/10 hover:border-primary/50 text-foreground/60 hover:text-foreground rounded-xl px-12 h-14 font-black uppercase tracking-widest text-[10px] transition-all shadow-lg cursor-pointer"
              >
                Link First Account
              </Button>
            </Card>
          ) : (
            accounts.map((acc, i) => (
              <Card
                key={i}
                className="liquid-glass rounded-xl hover:border-primary/40 transition-all group p-10 border-none shadow-inner"
              >
                <div className="flex items-center justify-between mb-10">
                  <div className="flex items-center gap-6">
                    <div className="w-14 h-14 rounded-xl bg-white/5 border-[0.5px] border-white/10 flex items-center justify-center overflow-hidden shrink-0 shadow-sm group-hover:border-primary/30 transition-all">
                      <span className="text-foreground font-black text-2xl group-hover:text-primary transition-colors italic">
                        {acc.bank_name?.charAt(0) || "B"}
                      </span>
                    </div>
                    <div>
                      <h4 className="text-foreground font-black text-xl tracking-tighter uppercase italic">
                        {acc.bank_name}
                      </h4>
                      <p className="text-foreground/30 text-[9px] font-mono tracking-[0.3em] mt-2 uppercase">
                        Account No: *{acc.account_number?.slice(-4) || "0000"}
                      </p>
                    </div>
                  </div>
                  <div className="bg-green-500/5 text-green-500 border-[0.5px] border-green-500/20 px-3 py-1 font-black uppercase tracking-widest flex items-center justify-center h-6 text-[8px] shadow-sm rounded-md">
                    Sync Active
                  </div>
                </div>

                <div className="grid grid-cols-2 gap-6 mb-10">
                  <div className="p-6 rounded-2xl bg-white/2 border-[0.5px] border-white/10 backdrop-blur-xl">
                    <p className="text-[8px] text-foreground/20 uppercase font-black tracking-[0.3em] mb-2">
                      Last synced
                    </p>
                    <p className="text-foreground text-[10px] font-bold font-mono tracking-widest uppercase">
                      2 min ago
                    </p>
                  </div>
                  <div className="p-6 rounded-2xl bg-white/2 border-[0.5px] border-white/10 backdrop-blur-xl">
                    <p className="text-[8px] text-foreground/20 uppercase font-black tracking-[0.3em] mb-2">
                      Expiry
                    </p>
                    <p className="text-foreground text-[10px] font-bold font-mono tracking-widest uppercase">
                      84 Days
                    </p>
                  </div>
                </div>

                <div className="flex items-center gap-4">
                  <Button
                    onPress={handleConnect}
                    className="flex-1 flex items-center justify-center bg-white/5 hover:bg-white/10 text-foreground/60 hover:text-foreground border-[0.5px] border-white/10 rounded-xl h-12 font-black text-[9px] uppercase tracking-widest transition-all cursor-pointer shadow-sm"
                  >
                    <RefreshCcw size={14} className="mr-3" /> Reconnect account
                  </Button>
                  <Button
                    isIconOnly
                    className="w-12 min-w-12 h-12 bg-primary/5 hover:bg-primary/20 text-primary border-[0.5px] border-primary/20 rounded-xl flex items-center justify-center transition-all cursor-pointer shadow-sm"
                  >
                    <Trash2 size={18} />
                  </Button>
                </div>
              </Card>
            ))
          )}
        </div>

        <div className="space-y-8">
          <h3 className="text-[10px] font-black text-foreground/30 uppercase tracking-[0.4em] px-2 italic">
            Security Infrastructure
          </h3>
          <Card className="liquid-glass rounded-xl p-10 border-none shadow-inner">
            <div className="flex items-center gap-5 mb-12">
              <div className="w-10 h-10 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shrink-0 shadow-sm">
                <ShieldCheck size={20} />
              </div>
              <div>
                <h4 className="text-foreground font-black text-lg tracking-widest uppercase italic m-0">
                  Data Privacy
                </h4>
                <p className="text-foreground/30 text-[9px] font-black uppercase tracking-[0.4em] mt-1">
                  Institutional Standard
                </p>
              </div>
            </div>

            <div className="space-y-4">
              {[
                { label: "Institutional Data Streams" },
                { label: "Real-time Balance States" },
                { label: "Direct Debit Analysis" },
                { label: "Secure Identity Verification" },
              ].map((item, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between p-5 rounded-xl bg-white/1 border-[0.5px] border-white/5 hover:bg-white/3 transition-all group"
                >
                  <span className="text-[11px] text-foreground/60 font-bold uppercase tracking-widest group-hover:text-foreground transition-colors">
                    {item.label}
                  </span>
                  <CheckCircle2 size={16} className="text-green-500/40" />
                </div>
              ))}
            </div>

            <div className="mt-12 p-8 rounded-2xl bg-primary/5 border-[0.5px] border-primary/20 flex items-center gap-6 backdrop-blur-xl">
              <div className="w-10 h-10 rounded-xl bg-primary/10 flex items-center justify-center shrink-0 border-[0.5px] border-primary/20">
                <AlertTriangle size={20} className="text-primary" />
              </div>
              <div>
                <p className="text-[11px] text-foreground font-black mb-1 uppercase tracking-widest">
                  Connection Status Notice
                </p>
                <p className="text-[9px] text-foreground/40 leading-relaxed font-bold uppercase tracking-tight">
                  90-day reconnection required for your security. You will be
                  notified 7 days before expiration.
                </p>
              </div>
            </div>
          </Card>
        </div>
      </div>
    </div>
  );
}
