import DashboardClient from "./DashboardClient";
import CashFlowWidget from "@/app/(protected)/_components/widgets/Cashflow";
import SpendingTrendWidget from "@/app/(protected)/_components/widgets/Trends";
import ExpenseDistributionWidget from "@/app/(protected)/_components/widgets/Categories";
import PortfolioCardWidget from "@/app/(protected)/_components/widgets/ConnectedAccounts";
import CommodityMarketWidget from "@/app/(protected)/_components/widgets/Markets";
import FinancialNewsWidget from "@/app/(protected)/_components/widgets/News";
import BalanceForecastWidget from "@/app/(protected)/_components/widgets/BalanceForecast";
import MarketTicker from "@/app/(protected)/_components/MarketTicker";
import { Habits } from "@/app/(protected)/_components/widgets/analytics/Habits";
import { Subscriptions } from "@/app/(protected)/_components/widgets/analytics/Subscriptions";
import { Anomalies } from "@/app/(protected)/_components/widgets/analytics/Anomalies";
import { Risk } from "@/app/(protected)/_components/widgets/analytics/Risk";

import { Health } from "@/app/(protected)/_components/widgets/analytics/Health";
import LedgerTableWidget from "@/app/(protected)/_components/widgets/Transactions";

import GoalsWidget from "@/app/(protected)/_components/widgets/Goals/client";
import RecurringWidget from "@/app/(protected)/_components/widgets/Recurring/client";
import DebtWidget from "@/app/(protected)/_components/widgets/Debt/client";


export default async function HomePage() {
  const widgetsMap = {
    cashFlow: <CashFlowWidget />,
    spendingTrend: <SpendingTrendWidget />,
    expenseDistribution: <ExpenseDistributionWidget />,
    portfolio: <PortfolioCardWidget />,
    commodityMarket: <CommodityMarketWidget />,
    financialNews: <FinancialNewsWidget />,
    balanceForecast: <BalanceForecastWidget />,
    analyticsHabits: <Habits />,
    analyticsSubscriptions: <Subscriptions />,
    analyticsAnomalies: <Anomalies />,
    analyticsRisk: <Risk />,
    analyticsHealth: <Health />,
    ledger: <LedgerTableWidget />,

    goalsProgress: <GoalsWidget />,
    recurringSubs: <RecurringWidget />,
    debtLiabilities: <DebtWidget />,
  };

  // Sever-Side Initial Paint Fetching
  let initialBuckets = [];
  let initialAlerts = [];
  try {
    const API_BASE_URL = process.env.API_BASE_URL || process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";
    // We try to fetch the initial data for hydration
    const [bucketsRes, alertsRes] = await Promise.all([
      fetch(`${API_BASE_URL}/api/buckets`, { cache: "no-store" }).catch(() => null),
      fetch(`${API_BASE_URL}/api/alerts/history`, { cache: "no-store" }).catch(() => null)
    ]);
    
    if (bucketsRes?.ok) {
      const data = await bucketsRes.json();
      initialBuckets = data.buckets || [];
    } else {
      // Mock Data for presentation if backend endpoint doesn't exist yet
      initialBuckets = [
        { type: "UNALLOCATED", title: "Unallocated Funds", balance: 2450.00, sparklineData: [40, 50, 45, 60, 55, 70], progress: undefined }
      ];
    }

    if (alertsRes?.ok) {
      initialAlerts = await alertsRes.json();
    }
  } catch (err) {
    console.error("Failed to fetch initial server state", err);
  }

  return <DashboardClient 
    widgetsMap={widgetsMap} 
    ticker={<MarketTicker />} 
    initialBuckets={initialBuckets}
    initialAlerts={initialAlerts}
  />;
}
