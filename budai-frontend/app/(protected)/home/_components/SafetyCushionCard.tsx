"use client";

import React, { useState, useEffect, useMemo } from "react";
import { Button } from "@heroui/react";
import {
  ShieldCheck,
  SlidersHorizontal,
  Check,
  X,
  Lock,
  AlertCircle,
  PauseCircle,
  PlayCircle,
} from "lucide-react";
import { formatCurrency, getCurrencySymbol } from "@/lib/currency";
import { useBudAI } from "@/app/context/AppContext";
import { toast } from "@heroui/react";

interface SafetyCushionCardProps {
  currency?: string;
}

export const SafetyCushionCard = ({
  currency = "GBP",
}: SafetyCushionCardProps) => {
  const { snapshot, activeAccountId, accounts } = useBudAI();

  const [safetyCushion, setSafetyCushion] = useState<number>(500);
  const [dailyLimit, setDailyLimit] = useState<number>(250);
  const [isAutoEnabled, setIsAutoEnabled] = useState<boolean>(true);
  const [isEditing, setIsEditing] = useState<boolean>(false);

  const [editCushion, setEditCushion] = useState<string>("500");
  const [editLimit, setEditLimit] = useState<string>("250");

  useEffect(() => {
    if (typeof window !== "undefined") {
      const savedCushion = localStorage.getItem("budai_safety_cushion");
      if (savedCushion) {
        const val = parseFloat(savedCushion);
        if (!isNaN(val)) {
          setSafetyCushion(val);
          setEditCushion(String(val));
        }
      }

      const savedLimit = localStorage.getItem("budai_daily_limit");
      if (savedLimit) {
        const val = parseFloat(savedLimit);
        if (!isNaN(val)) {
          setDailyLimit(val);
          setEditLimit(String(val));
        }
      }

      const savedAuto = localStorage.getItem("budai_auto_moves_enabled");
      if (savedAuto !== null) {
        setIsAutoEnabled(savedAuto === "true");
      }
    }
  }, []);

  const activeCurrency = useMemo(() => {
    if (activeAccountId && activeAccountId !== "ALL") {
      const acc = accounts.find((a) => a.account_id === activeAccountId);
      if (acc?.currency) return acc.currency;
    }
    return accounts[0]?.currency || currency;
  }, [accounts, activeAccountId, currency]);

  const sym = getCurrencySymbol(activeCurrency);

  // Total liquid cash balance from snapshot
  const currentBalance = useMemo(() => {
    if (!snapshot) return 0;
    if (typeof snapshot.financial_anchor?.total_physical_cash === "number") {
      return snapshot.financial_anchor.total_physical_cash;
    }
    if (typeof snapshot.buckets?.unallocated?.cached_balance === "number") {
      return snapshot.buckets.unallocated.cached_balance;
    }
    return 0;
  }, [snapshot]);

  // Buffer calculation
  const buffer = currentBalance - safetyCushion;
  const isSafe = buffer >= 0;

  // Safe meter bar proportions
  const meterWidths = useMemo(() => {
    if (currentBalance <= 0) {
      return { protectedPct: 100, safePct: 0, isDeficit: true };
    }
    if (currentBalance < safetyCushion) {
      const currentPct = Math.max(8, Math.round((currentBalance / safetyCushion) * 100));
      return { protectedPct: currentPct, safePct: 0, isDeficit: true };
    }
    const protectedRatio = Math.round((safetyCushion / currentBalance) * 100);
    const safeRatio = Math.max(0, 100 - protectedRatio);
    return {
      protectedPct: Math.max(12, Math.min(88, protectedRatio)),
      safePct: Math.max(12, Math.min(88, safeRatio)),
      isDeficit: false,
    };
  }, [currentBalance, safetyCushion]);

  // Daily moves today (placeholder calculated from snapshot transfer items if available)
  const movesToday = useMemo(() => {
    return 0;
  }, []);

  const dailyUsedPct = Math.min(
    100,
    dailyLimit > 0 ? Math.round((movesToday / dailyLimit) * 100) : 0
  );

  const handleSave = () => {
    const parsedCushion = parseFloat(editCushion);
    const parsedLimit = parseFloat(editLimit);

    if (isNaN(parsedCushion) || parsedCushion < 0) {
      toast("Please enter a valid minimum balance.");
      return;
    }
    if (isNaN(parsedLimit) || parsedLimit < 0) {
      toast("Please enter a valid daily limit.");
      return;
    }

    setSafetyCushion(parsedCushion);
    setDailyLimit(parsedLimit);

    localStorage.setItem("budai_safety_cushion", String(parsedCushion));
    localStorage.setItem("budai_daily_limit", String(parsedLimit));

    setIsEditing(false);
    toast("Safety limits saved successfully.");
  };

  const handleToggleAuto = () => {
    const nextState = !isAutoEnabled;
    setIsAutoEnabled(nextState);
    localStorage.setItem("budai_auto_moves_enabled", String(nextState));
    if (nextState) {
      toast("Automatic moves turned on.");
    } else {
      toast("Automatic moves paused. BudAI will ask before moving money.");
    }
  };

  return (
    <div className="p-5 rounded-2xl bg-[#0d0e14]/40 backdrop-blur-xl border border-white/10 shadow-xl flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div className="flex items-center gap-2.5">
          <div className="p-1.5 rounded-xl bg-emerald-500/10 text-emerald-500 border border-emerald-500/20">
            <ShieldCheck size={18} />
          </div>
          <div>
            <h3 className="text-sm font-bold text-foreground">Safety Cushion</h3>
            <p className="text-[11px] text-muted-foreground">
              Balance protection and transfer limits
            </p>
          </div>
        </div>

        <div className="flex items-center gap-2">
          <button
            onClick={handleToggleAuto}
            className={`inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full text-xs font-semibold border transition-all cursor-pointer ${
              isAutoEnabled
                ? "bg-emerald-500/15 text-emerald-600 dark:text-emerald-400 border-emerald-500/25 hover:bg-emerald-500/25"
                : "bg-amber-500/15 text-amber-600 dark:text-amber-400 border-amber-500/25 hover:bg-amber-500/25"
            }`}
            title="Click to toggle automatic moves"
          >
            <span
              className={`w-1.5 h-1.5 rounded-full ${
                isAutoEnabled ? "bg-emerald-500 animate-pulse" : "bg-amber-500"
              }`}
            />
            <span>{isAutoEnabled ? "Active" : "Paused"}</span>
          </button>

          <button
            onClick={() => {
              setEditCushion(String(safetyCushion));
              setEditLimit(String(dailyLimit));
              setIsEditing(!isEditing);
            }}
            aria-label="Adjust limits"
            className="p-1.5 text-muted-foreground hover:text-foreground rounded-lg hover:bg-white/5 border border-white/10 transition-colors cursor-pointer"
          >
            <SlidersHorizontal size={14} />
          </button>
        </div>
      </div>

      {/* Inline Adjuster Mode */}
      {isEditing ? (
        <div className="p-4 rounded-xl bg-white/5 border border-white/10 flex flex-col gap-3.5 animate-in fade-in duration-200">
          <div className="flex items-center justify-between pb-2 border-b border-white/10">
            <span className="text-xs font-bold text-foreground">Adjust Limits</span>
            <button
              onClick={() => setIsEditing(false)}
              className="text-muted-foreground hover:text-foreground cursor-pointer"
            >
              <X size={14} />
            </button>
          </div>

          {/* Edit Minimum Balance */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-medium text-muted-foreground">
              Minimum balance to keep untouched:
            </label>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1 flex-1 px-2.5 py-1.5 bg-black/40 border border-white/15 rounded-xl text-xs font-mono font-bold text-foreground focus-within:border-primary/50">
                <span className="text-muted-foreground">{sym}</span>
                <input
                  type="number"
                  min="0"
                  step="50"
                  value={editCushion}
                  onChange={(e) => setEditCushion(e.target.value)}
                  className="w-full bg-transparent outline-none font-mono"
                  placeholder="500"
                />
              </div>
              <div className="flex items-center gap-1">
                {[250, 500, 1000].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setEditCushion(String(preset))}
                    className="px-2 py-1 text-[10px] font-semibold rounded-lg bg-white/5 hover:bg-white/15 border border-white/10 text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                  >
                    {sym}{preset}
                  </button>
                ))}
              </div>
            </div>
          </div>

          {/* Edit Daily Limit */}
          <div className="flex flex-col gap-1.5">
            <label className="text-[11px] font-medium text-muted-foreground">
              Maximum money BudAI can move in a day:
            </label>
            <div className="flex items-center gap-2">
              <div className="flex items-center gap-1 flex-1 px-2.5 py-1.5 bg-black/40 border border-white/15 rounded-xl text-xs font-mono font-bold text-foreground focus-within:border-primary/50">
                <span className="text-muted-foreground">{sym}</span>
                <input
                  type="number"
                  min="0"
                  step="50"
                  value={editLimit}
                  onChange={(e) => setEditLimit(e.target.value)}
                  className="w-full bg-transparent outline-none font-mono"
                  placeholder="250"
                />
              </div>
              <div className="flex items-center gap-1">
                {[100, 250, 500].map((preset) => (
                  <button
                    key={preset}
                    type="button"
                    onClick={() => setEditLimit(String(preset))}
                    className="px-2 py-1 text-[10px] font-semibold rounded-lg bg-white/5 hover:bg-white/15 border border-white/10 text-muted-foreground hover:text-foreground transition-colors cursor-pointer"
                  >
                    {sym}{preset}
                  </button>
                ))}
              </div>
            </div>
          </div>

          <div className="flex items-center justify-end gap-2 pt-2 border-t border-white/10">
            <Button
              size="sm"
              variant="ghost"
              onPress={() => setIsEditing(false)}
              className="text-xs h-7 px-3 rounded-lg"
            >
              Cancel
            </Button>
            <Button
              size="sm"
              variant="primary"
              onPress={handleSave}
              className="text-xs h-7 px-3 rounded-lg font-semibold"
            >
              Save Limits
            </Button>
          </div>
        </div>
      ) : (
        /* Normal View: Gauges & Metrics */
        <div className="flex flex-col gap-3.5">
          {/* Gauge 1: Minimum Balance & Safety Meter */}
          <div className="flex flex-col gap-2 p-3 rounded-xl bg-white/[0.03] border border-white/10">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground font-medium">Minimum Balance</span>
              <span className="font-mono font-bold text-foreground">
                {formatCurrency(safetyCushion, activeCurrency)}
              </span>
            </div>

            {/* Visual Safety Meter Bar */}
            <div className="w-full h-3 rounded-full bg-white/5 border border-white/10 overflow-hidden flex p-0.5 gap-0.5">
              <div
                style={{ width: `${meterWidths.protectedPct}%` }}
                className={`h-full rounded-full transition-all duration-500 ${
                  meterWidths.isDeficit
                    ? "bg-amber-500/80"
                    : "bg-neutral-600/60 dark:bg-neutral-500/50"
                }`}
                title={`Protected Minimum: ${formatCurrency(safetyCushion, activeCurrency)}`}
              />
              {!meterWidths.isDeficit && (
                <div
                  style={{ width: `${meterWidths.safePct}%` }}
                  className="h-full rounded-full bg-emerald-500/80 shadow-[0_0_8px_rgba(16,185,129,0.4)] transition-all duration-500"
                  title={`Safe Cushion: ${formatCurrency(Math.max(0, buffer), activeCurrency)}`}
                />
              )}
            </div>

            {/* Meter Legend & Live Status */}
            <div className="flex items-center justify-between text-[11px] pt-0.5">
              <div className="flex items-center gap-1.5">
                <span
                  className={`w-1.5 h-1.5 rounded-full ${
                    isSafe ? "bg-emerald-500" : "bg-amber-500"
                  }`}
                />
                <span
                  className={`font-medium ${
                    isSafe
                      ? "text-emerald-500 dark:text-emerald-400"
                      : "text-amber-500 dark:text-amber-400"
                  }`}
                >
                  {isSafe
                    ? `Safe: ${formatCurrency(buffer, activeCurrency)} above minimum`
                    : `Low: ${formatCurrency(Math.abs(buffer), activeCurrency)} below minimum`}
                </span>
              </div>
              <span className="text-muted-foreground text-[10px]">
                Balance: {formatCurrency(currentBalance, activeCurrency)}
              </span>
            </div>
          </div>

          {/* Gauge 2: Max Moved Per Day */}
          <div className="flex flex-col gap-2 p-3 rounded-xl bg-white/[0.03] border border-white/10">
            <div className="flex items-center justify-between text-xs">
              <span className="text-muted-foreground font-medium">Max Moved Per Day</span>
              <span className="font-mono font-bold text-foreground">
                {formatCurrency(dailyLimit, activeCurrency)} / day
              </span>
            </div>

            {/* Daily Usage Progress Bar */}
            <div className="w-full h-2 rounded-full bg-white/5 border border-white/10 overflow-hidden">
              <div
                style={{ width: `${dailyUsedPct}%` }}
                className="h-full rounded-full bg-primary transition-all duration-500"
              />
            </div>

            <div className="flex items-center justify-between text-[11px] text-muted-foreground">
              <span>{formatCurrency(movesToday, activeCurrency)} moved today</span>
              <span>
                {formatCurrency(Math.max(0, dailyLimit - movesToday), activeCurrency)} left
              </span>
            </div>
          </div>
        </div>
      )}

      {/* Safety Note Footer */}
      <div className="flex items-center gap-2 pt-2 border-t border-white/10 text-[11px] text-muted-foreground/80 leading-relaxed">
        <Lock size={12} className="shrink-0 text-emerald-500/80" />
        <span>Read-only bank connection. Your real money is never moved.</span>
      </div>
    </div>
  );
};
