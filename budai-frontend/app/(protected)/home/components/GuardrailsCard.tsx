"use client";

import React from "react";
import { Card, Chip } from "@heroui/react";
import { Check } from "lucide-react";

interface GuardrailsData {
  safety_buffer?: number;
  autonomous_limit?: number;
  approval_policy?: string;
  bank_connection_protocol?: string;
  guardrails_active?: boolean;
}

interface GuardrailsCardProps {
  guardrails?: GuardrailsData;
}

export function GuardrailsCard({ guardrails }: GuardrailsCardProps) {
  const safetyBuffer = guardrails?.safety_buffer ?? 800.0;
  const autonomousLimit = guardrails?.autonomous_limit ?? 150.0;
  const approvalPolicy = guardrails?.approval_policy ?? "Strict (Two-Factor)";
  const bankProtocol = guardrails?.bank_connection_protocol ?? "Open Banking 2.1";

  return (
    <Card className="glass-card rounded-2xl p-4 border border-slate-800 shadow-none space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between pb-2 border-b border-slate-800/80 mb-1 w-full">
        <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
          Agent Guardrails
        </h2>
        <Chip size="sm" variant="soft" color="success" className="font-mono text-[11px]">
          <span className="w-1.5 h-1.5 rounded-full bg-emerald-400 inline-block mr-1" />
          All Active
        </Chip>
      </div>

      {/* Constraints Grid */}
      <div className="space-y-2 text-xs w-full">
        <div className="flex items-center justify-between p-2 rounded-lg bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400">Safety Buffer</span>
          <span className="font-mono font-semibold text-emerald-400">
            £{safetyBuffer.toFixed(2)} (Protected)
          </span>
        </div>

        <div className="flex items-center justify-between p-2 rounded-lg bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400">Autonomous Limit</span>
          <span className="font-mono text-slate-200">
            £{autonomousLimit.toFixed(2)} / action
          </span>
        </div>

        <div className="flex items-center justify-between p-2 rounded-lg bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400">Approval Policy</span>
          <span className="text-slate-300 font-medium">{approvalPolicy}</span>
        </div>

        <div className="flex items-center justify-between p-2 rounded-lg bg-slate-900/60 border border-slate-800">
          <span className="text-slate-400">Bank Connection</span>
          <span className="text-cyan-400 flex items-center gap-1 font-mono text-[11px]">
            <Check className="w-3 h-3 text-cyan-400" />
            {bankProtocol}
          </span>
        </div>
      </div>
    </Card>
  );
}
