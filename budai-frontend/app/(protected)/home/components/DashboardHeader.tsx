"use client";

import React from "react";
import { Card, Button, Chip } from "@heroui/react";
import { ShieldCheck } from "lucide-react";

interface ReconciliationData {
  physical_balance: number;
  virtual_balance: number;
  delta: number;
  is_reconciled: boolean;
  bank_synced: boolean;
}

interface DashboardHeaderProps {
  reconciliation?: ReconciliationData;
  timeWindow: string;
  onTimeWindowChange: (window: string) => void;
}

export function DashboardHeader({
  reconciliation,
  timeWindow,
  onTimeWindowChange,
}: DashboardHeaderProps) {
  const physical = reconciliation?.physical_balance ?? 4500.0;
  const isReconciled = reconciliation?.is_reconciled ?? true;
  const delta = reconciliation?.delta ?? 0.0;

  return (
    <Card className="glass-card rounded-2xl p-4 border border-slate-800 shadow-none">
      <div className="flex flex-wrap items-center justify-between gap-4 w-full">
        {/* Brand & System Title */}
        <div className="flex items-center space-x-3">
          <div className="w-9 h-9 rounded-xl bg-gradient-to-tr from-cyan-500 to-blue-600 flex items-center justify-center font-bold text-white shadow-md shadow-cyan-500/20">
            <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 10V3L4 14h7v7l9-11h-7z" />
            </svg>
          </div>
          <div>
            <div className="flex items-center space-x-2">
              <span className="text-base font-semibold text-white tracking-tight">BudAI</span>
              <Chip size="sm" variant="soft" color="accent" className="font-mono text-xs">
                Autonomous
              </Chip>
            </div>
            <p className="text-xs text-slate-400">Primary Treasury & Cash Management</p>
          </div>
        </div>

        {/* Global Time Window Filter */}
        <div className="flex items-center bg-slate-900/90 p-1 rounded-xl border border-slate-800">
          {["30D", "60D Forecast", "90D", "1Y"].map((windowOption) => {
            const isActive = timeWindow === windowOption;
            return (
              <Button
                key={windowOption}
                size="sm"
                variant={isActive ? "secondary" : "ghost"}
                onPress={() => onTimeWindowChange(windowOption)}
                className={`text-xs font-medium px-3 h-7 ${
                  isActive
                    ? "bg-slate-800 text-cyan-400 border border-slate-700/60"
                    : "text-slate-400 hover:text-white"
                }`}
              >
                {windowOption}
              </Button>
            );
          })}
        </div>

        {/* Zero-Sum Reconciliation & Guardrail Badges */}
        <div className="flex items-center flex-wrap gap-2">
          <Chip
            size="md"
            variant="soft"
            color="success"
            className="font-mono text-xs py-1.5 px-3 flex items-center gap-1.5"
          >
            <span className="w-2 h-2 rounded-full bg-emerald-400 animate-pulse inline-block mr-1" />
            <span>
              Cash: <strong>£{physical.toLocaleString("en-GB", { minimumFractionDigits: 2, maximumFractionDigits: 2 })}</strong>
            </span>
            <span className="text-emerald-500 mx-1">|</span>
            <span className="font-medium">
              {isReconciled ? "Reconciled (Delta £0.00)" : `Delta £${delta.toFixed(2)}`}
            </span>
          </Chip>

          <Chip
            size="md"
            variant="soft"
            color="accent"
            className="text-xs py-1.5 px-3 flex items-center gap-1.5"
          >
            <ShieldCheck className="w-3.5 h-3.5 text-cyan-400 inline-block mr-1" />
            <span className="font-medium text-cyan-200">Guardrails Active</span>
            <span className="text-slate-500 mx-1">•</span>
            <span className="text-cyan-400 font-mono">Autonomous</span>
          </Chip>
        </div>
      </div>
    </Card>
  );
}
