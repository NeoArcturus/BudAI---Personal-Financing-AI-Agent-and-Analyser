"use client";

import React, { useState } from "react";
import { ChevronDown, ArrowRightLeft, Check, Sparkles, Database, TrendingUp, ShieldAlert, Brain, FileSpreadsheet } from "lucide-react";
import { motion, AnimatePresence } from "framer-motion";
import { Chip } from "@heroui/react";
import { cn } from "@/lib/utils";

export const AGENT_DIRECTORY: Record<string, { name: string; role: string; icon: React.ComponentType<{ size?: number; className?: string }> }> = {
  call_analyser_agent: { name: "Financial Analyst", role: "Spending Patterns & Outliers", icon: TrendingUp },
  AnalyserAgent: { name: "Financial Analyst", role: "Spending Patterns & Outliers", icon: TrendingUp },
  analyser_worker: { name: "Financial Analyst", role: "Spending Patterns & Outliers", icon: TrendingUp },
  call_categorizer_agent: { name: "Transaction Classifier", role: "Category & Merchant Mapping", icon: Brain },
  CategorizerAgent: { name: "Transaction Classifier", role: "Category & Merchant Mapping", icon: Brain },
  call_market_agent: { name: "Market Intelligence", role: "Macroeconomic Context", icon: Sparkles },
  MarketAgent: { name: "Market Intelligence", role: "Macroeconomic Context", icon: Sparkles },
  call_health_agent: { name: "Financial Health Monitor", role: "Runway & Risk Audit", icon: ShieldAlert },
  HealthAgent: { name: "Financial Health Monitor", role: "Runway & Risk Audit", icon: ShieldAlert },
  call_scenario_agent: { name: "Scenario Simulator", role: "Forecast & Cashflow Modeling", icon: TrendingUp },
  ScenarioAgent: { name: "Scenario Simulator", role: "Forecast & Cashflow Modeling", icon: TrendingUp },
  call_memory_agent: { name: "Profile Memory", role: "User Goals & Directives", icon: Brain },
  MemoryAgent: { name: "Profile Memory", role: "User Goals & Directives", icon: Brain },
  query_transactions: { name: "Data Ingestion", role: "Transaction Record Retrieval", icon: Database },
  aggregate_financial_data: { name: "Metrics Aggregator", role: "Category & Monthly Summaries", icon: Database },
  get_liability_horizon: { name: "Liability Tracker", role: "Pending Debits & Subscriptions", icon: FileSpreadsheet },
  export_custom_statement: { name: "Statement Exporter", role: "CSV Statement Generation", icon: FileSpreadsheet },
};

export function resolveAgentMetadata(rawName: string) {
  if (AGENT_DIRECTORY[rawName]) {
    return AGENT_DIRECTORY[rawName];
  }
  const cleanName = rawName
    .replace(/^call_/, "")
    .replace(/_agent$/, "")
    .replace(/_/g, " ")
    .replace(/\b\w/g, (c) => c.toUpperCase());
  return {
    name: cleanName || "Specialist Agent",
    role: "Financial Task Delegation",
    icon: ArrowRightLeft,
  };
}

interface AgentHandoverCardProps {
  toolName: string;
  args: any;
  state?: "call" | "result";
  result?: any;
}

export const AgentHandoverCard: React.FC<AgentHandoverCardProps> = ({
  toolName,
  args,
  state = "result",
  result,
}) => {
  const [isExpanded, setIsExpanded] = useState(false);
  const metadata = resolveAgentMetadata(toolName);
  const IconComponent = metadata.icon;
  const isRunning = state === "call" && !result;

  // Format inquiry/handover information cleanly
  const handoverSummary = args?.query || (typeof args === "object" ? JSON.stringify(args, null, 2) : String(args || ""));

  // Format received result cleanly
  let receivedSummary = "";
  if (result) {
    if (typeof result === "string") {
      try {
        const parsed = JSON.parse(result);
        if (parsed.data && Array.isArray(parsed.data)) {
          receivedSummary = `${parsed.data.length} records returned. Status: ${parsed.status || "success"}`;
        } else {
          receivedSummary = JSON.stringify(parsed, null, 2);
        }
      } catch {
        receivedSummary = result;
      }
    } else if (typeof result === "object") {
      receivedSummary = JSON.stringify(result, null, 2);
    } else {
      receivedSummary = String(result);
    }
  }

  return (
    <div className="w-full my-2.5 rounded-xl border border-white/10 bg-black/40 backdrop-blur-md overflow-hidden transition-all duration-200">
      <div
        onClick={() => setIsExpanded((prev) => !prev)}
        className="w-full px-3.5 py-2.5 flex items-center justify-between text-left hover:bg-white/[0.04] transition-colors cursor-pointer select-none"
      >
        <div className="flex items-center gap-2.5 min-w-0">
          <div className="w-6 h-6 rounded-lg bg-primary/10 border border-primary/20 flex items-center justify-center shrink-0 text-primary">
            <IconComponent size={13} />
          </div>
          <div className="flex items-center gap-2 min-w-0">
            <span className="text-[11px] font-bold text-foreground/90 tracking-tight">
              {metadata.name}
            </span>
            <span className="hidden sm:inline-block text-[10px] text-foreground/40 font-mono">
              • {metadata.role}
            </span>
          </div>
        </div>

        <div className="flex items-center gap-2 shrink-0">
          <Chip
            size="sm"
            variant="soft"
            className={cn(
              "h-5 px-1.5 text-[9px] font-mono uppercase tracking-widest",
              isRunning
                ? "bg-primary/15 border border-primary/40 text-primary"
                : "bg-white/5 border border-white/10 text-foreground/60"
            )}
          >
            {isRunning ? (
              <span className="flex items-center gap-1">
                <span className="w-1.5 h-1.5 rounded-full bg-primary animate-ping" />
                <span>Running</span>
              </span>
            ) : (
              <span className="flex items-center gap-1">
                <Check size={10} className="text-success" />
                <span>Handover Complete</span>
              </span>
            )}
          </Chip>

          <motion.div animate={{ rotate: isExpanded ? 180 : 0 }} transition={{ duration: 0.2 }} className="text-foreground/40">
            <ChevronDown size={14} />
          </motion.div>
        </div>
      </div>

      <AnimatePresence initial={false}>
        {isExpanded && (
          <motion.div
            initial={{ height: 0, opacity: 0 }}
            animate={{ height: "auto", opacity: 1 }}
            exit={{ height: 0, opacity: 0 }}
            transition={{ duration: 0.2 }}
            className="border-t border-white/5 bg-black/60 px-4 py-3 text-xs font-mono"
          >
            <div className="space-y-3">
              <div>
                <span className="text-[9px] uppercase tracking-wider text-primary/70 font-bold block mb-1">
                  Information Handed Over
                </span>
                <div className="p-2.5 rounded-lg bg-white/[0.03] border border-white/5 text-[11px] text-foreground/80 whitespace-pre-wrap max-h-40 overflow-y-auto custom-scrollbar">
                  {handoverSummary || "No parameters specified."}
                </div>
              </div>

              {receivedSummary && (
                <div>
                  <span className="text-[9px] uppercase tracking-wider text-success/80 font-bold block mb-1">
                    Information Received Back
                  </span>
                  <div className="p-2.5 rounded-lg bg-white/[0.03] border border-white/5 text-[11px] text-foreground/80 whitespace-pre-wrap max-h-48 overflow-y-auto custom-scrollbar">
                    {receivedSummary}
                  </div>
                </div>
              )}
            </div>
          </motion.div>
        )}
      </AnimatePresence>
    </div>
  );
};
