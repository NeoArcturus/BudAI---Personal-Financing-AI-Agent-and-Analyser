// app/(protected)/home/page.tsx
"use client";

import React, { useState, useEffect } from "react";
import { useBudAI } from "@/app/context/AppContext";
import MoneyFlowWidget from "@/app/(protected)/_components/MoneyflowWidget";
import PortfolioCardWidget from "@/app/(protected)/_components/PortfolioCardWidget";
import LedgerTableWidget from "@/app/(protected)/_components/LedgerTableWidget";
import ExpenseBreakdownWidget from "@/app/(protected)/_components/ExpenseBreakdownWidget";
import { apiFetch } from "@/lib/api";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import {
  Search,
  Bell,
  Settings,
  LayoutDashboard,
  ArrowRightLeft,
  Clock,
  RefreshCcw,
  CreditCard,
  Moon,
  Sun,
} from "lucide-react";
import { NativeChartConfig, Transaction, ParsedCategory } from "@/types";

interface FlexibleDataPoint {
  Date?: string;
  date?: string;
  timestamp?: string;
  [key: string]: string | number | undefined;
}

interface BankExpensePayload {
  bank_name?: string;
  data: FlexibleDataPoint[];
}

export default function HomePage() {
  const {
    activeAccountId,
    setActiveAccountId,
    accounts,
    triggerExplanation,
    userName,
  } = useBudAI();

  const [localTransactions, setLocalTransactions] = useState<Transaction[]>([]);
  const [cashFlowConfig, setCashFlowConfig] =
    useState<NativeChartConfig | null>(null);
  const [expensesConfig, setExpensesConfig] =
    useState<NativeChartConfig | null>(null);
  const [categories, setCategories] = useState<ParsedCategory[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(true);

  const targetId =
    activeAccountId === "ALL" || !activeAccountId
      ? accounts.length > 0
        ? accounts[0].account_id
        : null
      : activeAccountId;

  useEffect(() => {
    let isMounted = true;

    const fetchDashboardData = async () => {
      if (!targetId) {
        setIsLoading(false);
        return;
      }

      setIsLoading(true);
      try {
        const txRes = await apiFetch(
          `/api/accounts/${targetId}/transactions`,
          {},
          true,
        );
        if (txRes.ok) {
          const txData = await txRes.json();
          if (isMounted)
            setLocalTransactions((txData.transactions as Transaction[]) || []);
        }

        const cfRes = await apiFetch(
          "/api/media/execute",
          {
            method: "POST",
            body: JSON.stringify({
              tool_name: "plot_cash_flow_mixed",
              parameters: { bank_name_or_id: targetId },
            }),
          },
          true,
        );
        const cfData = await cfRes.json();
        if (isMounted && (cfData.data || cfData)) {
          let payload = cfData.data || cfData;
          if (Array.isArray(payload)) {
            if (
              payload.length > 0 &&
              "data" in payload[0] &&
              Array.isArray((payload[0] as BankExpensePayload).data)
            ) {
              payload = (payload as BankExpensePayload[]).map((bank) => {
                if (bank.data && bank.data.length > 0) {
                  const firstItem = bank.data[0];
                  const dateKey =
                    firstItem.Date !== undefined
                      ? "Date"
                      : firstItem.date !== undefined
                        ? "date"
                        : "timestamp";

                  bank.data.sort((a, b) => {
                    const dateA = String(a[dateKey] || "");
                    const dateB = String(b[dateKey] || "");
                    return (
                      new Date(dateA).getTime() - new Date(dateB).getTime()
                    );
                  });
                }
                bank.data = bank.data.slice(-12);
                return bank;
              });
            } else if (payload.length > 0) {
              const flatPayload = payload as FlexibleDataPoint[];
              const firstItem = flatPayload[0];
              const dateKey =
                firstItem.Date !== undefined
                  ? "Date"
                  : firstItem.date !== undefined
                    ? "date"
                    : firstItem.timestamp !== undefined
                      ? "timestamp"
                      : null;

              if (dateKey) {
                flatPayload.sort((a, b) => {
                  const dateA = String(a[dateKey] || "");
                  const dateB = String(b[dateKey] || "");
                  return new Date(dateA).getTime() - new Date(dateB).getTime();
                });
              }
              payload = flatPayload.slice(-12);
            }
          }

          setCashFlowConfig(
            buildChartConfig(
              "cash_flow_mixed",
              payload,
              { bank_name_or_id: targetId },
              "",
            ),
          );
        }

        const expRes = await apiFetch(
          "/api/media/execute",
          {
            method: "POST",
            body: JSON.stringify({
              tool_name: "plot_expenses",
              parameters: {
                bank_name_or_id: targetId,
                plot_time_type: "Monthly",
              },
            }),
          },
          true,
        );
        const expData = await expRes.json();
        if (isMounted && (expData.data || expData)) {
          let payload = expData.data || expData;
          if (Array.isArray(payload)) {
            payload = (payload as BankExpensePayload[]).map((bank) => {
              if (bank.data && Array.isArray(bank.data)) {
                bank.data.sort((a, b) => {
                  const dateA = String(a.Date || a.date || a.timestamp || "");
                  const dateB = String(b.Date || b.date || b.timestamp || "");
                  return new Date(dateA).getTime() - new Date(dateB).getTime();
                });
                bank.data = bank.data.slice(-6);
              }
              return bank;
            });
          }
          setExpensesConfig(
            buildChartConfig(
              "historical_monthly",
              payload,
              { bank_name_or_id: targetId },
              "",
            ),
          );
        }

        const catRes = await apiFetch(
          "/api/media/execute",
          {
            method: "POST",
            body: JSON.stringify({
              tool_name: "classify_financial_data",
              parameters: { bank_name_or_id: targetId },
            }),
          },
          true,
        );
        const catData = await catRes.json();

        if (isMounted && (catData.data || catData)) {
          const payload = catData.data || catData;
          const bankDataArray = payload[0]?.data || [];
          const parsed: ParsedCategory[] = bankDataArray.map(
            (c: {
              Category?: string;
              category?: string;
              Total_Amount?: number | string;
              amount?: number | string;
            }) => ({
              name: c.Category || c.category || "Unknown",
              value: Number(c.Total_Amount) || Number(c.amount) || 0,
            }),
          );
          setCategories(parsed);
        }
      } catch (error) {
        console.log(error);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };

    fetchDashboardData();
    return () => {
      isMounted = false;
    };
  }, [targetId]);

  const handleFetchTransactions = async (from: string, to: string) => {
    if (!targetId) return;
    try {
      const response = await apiFetch(
        `/api/accounts/${targetId}/transactions?from=${from}&to=${to}`,
        {},
        true,
      );
      if (response.ok) {
        const data = await response.json();
        setLocalTransactions((data.transactions as Transaction[]) || []);
      }
    } catch (error) {
      console.log(error);
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#08090D] font-sans overflow-hidden">
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[20%] -left-[10%] w-[60%] h-[60%] rounded-full bg-[#FF8A4C]/15 blur-[140px]"></div>
        <div className="absolute -bottom-[20%] -right-[10%] w-[60%] h-[60%] rounded-full bg-[#3D73FF]/15 blur-[140px]"></div>
      </div>

      <div className="relative z-10 w-64 h-full bg-[#13151D]/40 backdrop-blur-xl border-r border-white/5 flex flex-col justify-between py-8 px-6 shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.2)]">
        <div>
          <div className="flex items-center gap-3 mb-12">
            <div className="w-8 h-8 rounded-lg bg-linear-to-br from-[#3D73FF] to-[#00E096] flex items-center justify-center shadow-lg">
              <span className="text-white font-bold text-lg leading-none">
                B
              </span>
            </div>
            <h1 className="text-white text-2xl font-bold tracking-tight">
              BudAI
            </h1>
          </div>

          <nav className="space-y-2">
            <a
              href="#"
              className="flex items-center gap-4 text-white bg-white/10 px-4 py-3 rounded-xl shadow-inner transition-all border border-white/5"
            >
              <LayoutDashboard size={20} />
              <span className="font-medium text-sm">Dashboard</span>
            </a>
            <a
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <ArrowRightLeft size={20} />
              <span className="font-medium text-sm">Transactions</span>
            </a>
            <a
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <Clock size={20} />
              <span className="font-medium text-sm">History</span>
            </a>
            <a
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <RefreshCcw size={20} />
              <span className="font-medium text-sm">Exchange</span>
            </a>
            <a
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <CreditCard size={20} />
              <span className="font-medium text-sm">Payments</span>
            </a>
            <a
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <Bell size={20} />
              <span className="font-medium text-sm">Notifications</span>
            </a>
            <a
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all mt-4"
            >
              <Settings size={20} />
              <span className="font-medium text-sm">Settings</span>
            </a>
          </nav>
        </div>

        <div className="flex items-center gap-3 pt-6 border-t border-white/10 mt-8">
          <div className="w-10 h-10 rounded-full bg-linear-to-br from-indigo-500 via-purple-500 to-pink-500 shrink-0 shadow-lg border border-white/10"></div>
          <div className="overflow-hidden">
            <p className="text-white text-sm font-medium truncate">
              {userName}
            </p>
            <p className="text-[#5E6272] text-xs truncate">BudAI Member</p>
          </div>
        </div>
      </div>

      <div className="relative z-10 flex-1 flex flex-col p-8 h-full overflow-hidden">
        <div className="flex items-center justify-between mb-8 shrink-0">
          <div className="relative w-100">
            <input
              type="text"
              placeholder="Search ledger..."
              className="w-full bg-[#13151D]/40 backdrop-blur-xl border border-white/5 text-sm text-white rounded-xl pl-11 pr-4 py-3 focus:border-[#3D73FF]/50 focus:outline-none transition-colors placeholder:text-[#5E6272] shadow-lg"
            />
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4.5 h-4.5 text-[#5E6272]" />
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center bg-[#13151D]/40 backdrop-blur-xl border border-white/5 rounded-full p-1 shadow-lg">
              <button className="w-9 h-9 rounded-full bg-orange-500/20 text-orange-400 flex items-center justify-center transition-colors shadow-inner">
                <Moon size={16} />
              </button>
              <button className="w-9 h-9 rounded-full text-[#5E6272] hover:text-white flex items-center justify-center transition-colors">
                <Sun size={16} />
              </button>
            </div>
            <button className="w-11 h-11 rounded-full bg-[#13151D]/40 backdrop-blur-xl border border-white/5 flex items-center justify-center text-white shadow-lg relative">
              <Bell size={18} />
              <span className="absolute top-3 right-3 w-2 h-2 bg-red-500 rounded-full border border-[#13151D]"></span>
            </button>
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 shrink-0 mb-6">
          <div className="lg:col-span-2 h-80">
            <MoneyFlowWidget isLoading={isLoading} config={cashFlowConfig} />
          </div>
          <div className="lg:col-span-1 h-80">
            <PortfolioCardWidget
              accounts={accounts}
              activeAccountId={targetId}
              onAccountSelect={setActiveAccountId}
            />
          </div>
        </div>

        <div className="grid grid-cols-1 lg:grid-cols-3 gap-6 flex-1 min-h-0">
          <div className="lg:col-span-2 h-full min-h-0">
            <LedgerTableWidget
              transactions={localTransactions}
              activeAccountId={targetId}
              onFilter={handleFetchTransactions}
            />
          </div>
          <div className="lg:col-span-1 h-full min-h-0">
            <ExpenseBreakdownWidget
              isLoading={isLoading}
              config={expensesConfig}
              categories={categories}
              onAnalyze={() =>
                triggerExplanation("CHART", expensesConfig?.data || null)
              }
            />
          </div>
        </div>
      </div>
    </div>
  );
}
