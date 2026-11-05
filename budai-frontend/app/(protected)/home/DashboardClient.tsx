"use client";

import React, { useState, useEffect, useMemo, createContext } from "react";
import { useRouter } from "next/navigation";
import { useBudAI } from "@/app/context/AppContext";
import { VirtualBucketsPanel } from "./_components/VirtualBucketsPanel";
import { FinancialAnalysisCard } from "./_components/FinancialAnalysisCard";
import { RecentTransactionsCard } from "./_components/RecentTransactionsCard";
import { AutomatedActivityPanel } from "./_components/AutomatedActivityPanel";

export interface WidgetContextType {
  onRemove?: () => void;
  instanceId?: string;
  isCustomizing?: boolean;
}

export const WidgetContext = createContext<WidgetContextType>({
  onRemove: () => {},
  instanceId: "",
  isCustomizing: false,
});

interface DashboardClientProps {
  ticker?: React.ReactNode;
  widgetsMap?: Record<string, React.ReactNode>;
  initialBuckets?: any[];
  initialAlerts?: any[];
}

export default function DashboardClient({
  ticker,
  widgetsMap = {},
  initialBuckets = [],
  initialAlerts = [],
}: DashboardClientProps) {
  const { userName, accounts, activeAccountId } = useBudAI();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // First time user check: If user has 0 linked accounts, redirect straight to onboarding
  useEffect(() => {
    if (mounted && accounts && accounts.length === 0) {
      const timer = setTimeout(() => {
        router.push("/onboarding");
      }, 500);
      return () => clearTimeout(timer);
    }
  }, [mounted, accounts, router]);

  const activeCurrency = useMemo(() => {
    if (activeAccountId && activeAccountId !== "ALL") {
      const acc = accounts.find((a) => a.account_id === activeAccountId);
      if (acc?.currency) return acc.currency;
    }
    return accounts[0]?.currency || "GBP";
  }, [accounts, activeAccountId]);

  if (!mounted) {
    return (
      <div className="w-full h-full flex items-center justify-center">
        <div className="animate-spin rounded-full h-8 w-8 border-2 border-primary border-t-transparent" />
      </div>
    );
  }

  return (
    <div className="w-full h-full overflow-y-auto scrollbar-hide px-4 sm:px-6 lg:px-8 xl:px-12 py-6 pb-28">
      <div className="w-full max-w-[1800px] mx-auto flex flex-col gap-6">
        {ticker && <div className="w-full shrink-0">{ticker}</div>}

        <div className="flex flex-col lg:flex-row items-start gap-6 w-full">
          {/* Left Column (Money Buckets) */}
          <div className="w-full lg:w-[320px] xl:w-[360px] 2xl:w-[400px] shrink-0">
            <VirtualBucketsPanel currency={activeCurrency} />
          </div>

          {/* Center Column (Financial Analysis & Transactions) - Un-squeezed & expansive */}
          <div className="w-full lg:flex-1 min-w-0 flex flex-col gap-6">
            <FinancialAnalysisCard currency={activeCurrency} />
            <RecentTransactionsCard currency={activeCurrency} />
          </div>

          {/* Right Column (Activity & Safety Safeguards) */}
          <div className="w-full lg:w-[320px] xl:w-[360px] 2xl:w-[400px] shrink-0">
            <AutomatedActivityPanel currency={activeCurrency} />
          </div>
        </div>
      </div>
    </div>
  );
}
