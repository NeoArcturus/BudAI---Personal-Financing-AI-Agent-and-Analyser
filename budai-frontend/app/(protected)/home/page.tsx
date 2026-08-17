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
  };

  return <DashboardClient widgetsMap={widgetsMap} ticker={<MarketTicker />} />;
}
