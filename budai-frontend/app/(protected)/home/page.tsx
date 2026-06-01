import DashboardClient from "./DashboardClient";
import CashFlowWidget from "@/app/(protected)/_components/CashFlowWidget";
import SpendingTrendWidget from "@/app/(protected)/_components/SpendingTrendWidget";
import ExpenseDistributionWidget from "@/app/(protected)/_components/ExpenseDistributionWidget";
import PortfolioCardWidget from "@/app/(protected)/_components/PortfolioCardWidget";
import LedgerTableWidget from "@/app/(protected)/_components/LedgerTableWidget";
import CommodityMarketWidget from "@/app/(protected)/_components/CommodityMarketWidget";
import FinancialNewsWidget from "@/app/(protected)/_components/FinancialNewsWidget";
import MarketTicker from "@/app/(protected)/_components/MarketTicker";

export default async function HomePage() {
  const widgetsMap = {
    cashFlow: <CashFlowWidget />,
    spendingTrend: <SpendingTrendWidget />,
    expenseDistribution: <ExpenseDistributionWidget />,
    portfolio: <PortfolioCardWidget />,
    ledger: <LedgerTableWidget />,
    commodityMarket: <CommodityMarketWidget />,
    financialNews: <FinancialNewsWidget />,
  };

  return <DashboardClient widgetsMap={widgetsMap} ticker={<MarketTicker />} />;
}
