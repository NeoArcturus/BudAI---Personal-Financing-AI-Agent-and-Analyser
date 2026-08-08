import DashboardClient from "./DashboardClient";
import CashFlowWidget from "@/app/(protected)/_components/widgets/Cashflow";
import SpendingTrendWidget from "@/app/(protected)/_components/widgets/Trends";
import ExpenseDistributionWidget from "@/app/(protected)/_components/widgets/Categories";
import PortfolioCardWidget from "@/app/(protected)/_components/widgets/ConnectedAccounts";
import LedgerTableWidget from "@/app/(protected)/_components/widgets/Transactions";
import CommodityMarketWidget from "@/app/(protected)/_components/widgets/Markets";
import FinancialNewsWidget from "@/app/(protected)/_components/widgets/News";
import MarketTicker from "@/app/(protected)/_components/MarketTicker";
import { Habits } from "@/app/(protected)/_components/widgets/analytics/Habits";
import { Subscriptions } from "@/app/(protected)/_components/widgets/analytics/Subscriptions";
import { Anomalies } from "@/app/(protected)/_components/widgets/analytics/Anomalies";
import { Risk } from "@/app/(protected)/_components/widgets/analytics/Risk";
import { Forecast } from "@/app/(protected)/_components/widgets/analytics/Forecast";
import { Health } from "@/app/(protected)/_components/widgets/analytics/Health";

export default async function HomePage() {
  const widgetsMap = {
    cashFlow: <CashFlowWidget />,
    spendingTrend: <SpendingTrendWidget />,
    expenseDistribution: <ExpenseDistributionWidget />,
    portfolio: <PortfolioCardWidget />,
    ledger: <LedgerTableWidget />,
    commodityMarket: <CommodityMarketWidget />,
    financialNews: <FinancialNewsWidget />,
    analyticsHabits: <Habits />,
    analyticsSubscriptions: <Subscriptions />,
    analyticsAnomalies: <Anomalies />,
    analyticsRisk: <Risk />,
    analyticsForecast: <Forecast />,
    analyticsHealth: <Health />,
  };

  return <DashboardClient widgetsMap={widgetsMap} ticker={<MarketTicker />} />;
}
