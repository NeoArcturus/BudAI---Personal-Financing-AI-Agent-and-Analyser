"use client";

import React from "react";
import { Card, Chip, Button } from "@heroui/react";

export interface DecisionItem {
  id: string;
  timestamp: string;
  status: "Executed" | "Awaiting Input" | string;
  title: string;
  observation: string;
  policy: string;
  action: string;
  requires_action: boolean;
}

interface DecisionStreamProps {
  decisions: DecisionItem[];
  onConfirmAction?: (id: string) => void;
  onReportAction?: (id: string) => void;
}

export function DecisionStream({
  decisions = [],
  onConfirmAction,
  onReportAction,
}: DecisionStreamProps) {
  return (
    <Card className="glass-card rounded-2xl p-4 border border-slate-800 shadow-none space-y-3">
      {/* Header */}
      <div className="flex items-center justify-between pb-3 border-b border-slate-800/80 mb-1 w-full">
        <div>
          <h2 className="text-xs font-semibold uppercase tracking-wider text-slate-400">
            Agent Decision Stream
          </h2>
          <p className="text-xs text-slate-500">Autonomous actions & causal rationale</p>
        </div>
        <span className="relative flex h-2 w-2">
          <span className="animate-ping absolute inline-flex h-full w-full rounded-full bg-cyan-400 opacity-75" />
          <span className="relative inline-flex rounded-full h-2 w-2 bg-cyan-500" />
        </span>
      </div>

      {/* Decision Cards */}
      <div className="space-y-3 w-full">
        {decisions.map((dec) => {
          const isPending = dec.requires_action || dec.status === "Awaiting Input";

          if (isPending) {
            return (
              <Card
                key={dec.id}
                className="p-3 rounded-xl bg-amber-950/20 border border-amber-800/40 shadow-none space-y-2"
              >
                <div className="flex items-center justify-between text-[11px] mb-1">
                  <span className="font-mono text-amber-400">{dec.timestamp}</span>
                  <Chip size="sm" variant="soft" color="warning" className="font-mono text-[10px]">
                    Awaiting Input
                  </Chip>
                </div>
                <p className="text-xs font-medium text-amber-200">{dec.title}</p>
                <div className="text-[11px] text-slate-300 space-y-1 bg-slate-950/60 p-2 rounded-lg border border-slate-800/60 font-mono">
                  <p>{dec.observation}</p>
                  <div className="flex gap-2 pt-1">
                    <Button
                      size="sm"
                      variant="primary"
                      className="bg-cyan-600 hover:bg-cyan-500 text-white rounded text-[11px] font-medium h-7 px-3"
                      onPress={() => onConfirmAction?.(dec.id)}
                    >
                      Confirm
                    </Button>
                    <Button
                      size="sm"
                      variant="secondary"
                      className="bg-slate-800 hover:bg-slate-700 text-slate-300 rounded text-[11px] h-7 px-3 border border-slate-700"
                      onPress={() => onReportAction?.(dec.id)}
                    >
                      Report
                    </Button>
                  </div>
                </div>
              </Card>
            );
          }

          return (
            <Card
              key={dec.id}
              className="p-3 rounded-xl bg-slate-900/80 border border-slate-800 shadow-none space-y-1.5"
            >
              <div className="flex items-center justify-between text-[11px] mb-1">
                <span className="font-mono text-cyan-400">{dec.timestamp}</span>
                <Chip size="sm" variant="soft" color="success" className="font-mono text-[10px]">
                  Executed
                </Chip>
              </div>
              <p className="text-xs font-medium text-white">{dec.title}</p>
              <div className="text-[11px] text-slate-400 space-y-1 bg-slate-950/60 p-2 rounded-lg border border-slate-800/60 font-mono">
                <p>
                  <strong className="text-slate-300">Observation:</strong> {dec.observation}
                </p>
                <p>
                  <strong className="text-slate-300">Policy:</strong> {dec.policy}
                </p>
                <p>
                  <strong className="text-cyan-400">Action:</strong> {dec.action}
                </p>
              </div>
            </Card>
          );
        })}
      </div>
    </Card>
  );
}
