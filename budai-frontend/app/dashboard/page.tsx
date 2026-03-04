"use client";

import React, { useState, useEffect } from "react";
import Image from "next/image";
import {
  CreditCard,
  Globe,
  History,
  X,
  BarChart3,
  ShieldCheck,
  TrendingUp,
  TrendingDown,
  Plus,
} from "lucide-react";
import BudAIChat from "./_components/BudAIChat";
import TransactionFeed from "./_components/TransactionFeed";
import { Account, Transaction } from "@/types";

interface MediaCache {
  categorized: Transaction[] | null;
  balanceForecastUrl: string;
  expenseForecastUrl: string;
}

type TabType = "raw" | "categorized" | "balance_forecast" | "expense_forecast";

export default function Dashboard() {
  const [accounts, setAccounts] = useState<Account[]>([]);
  // Tracks the ID of the currently selected account for the AI's context
  const [activeAccountId, setActiveAccountId] = useState<string | null>(null);
  const [displayMode, setDisplayMode] = useState<
    "commodities" | "account_details"
  >("commodities");
  const [activeTab, setActiveTab] = useState<TabType>("raw");

  const [selectedAccTransactions, setSelectedAccTransactions] = useState<
    Transaction[] | null
  >(null);
  const [mediaCache, setMediaCache] = useState<MediaCache>({
    categorized: null,
    balanceForecastUrl: "",
    expenseForecastUrl: "",
  });

  useEffect(() => {
    fetch("http://localhost:8080/api/accounts")
      .then((res) => res.json())
      .then((data) => setAccounts(data.accounts));
  }, []);

  const fetchAccountDetails = async (accountId: string) => {
    setActiveAccountId(accountId);
    const timestamp = new Date().getTime();

    const res = await fetch(
      `http://localhost:8080/api/accounts/${accountId}/transactions`,
    );
    const data = await res.json();
    setSelectedAccTransactions(data.transactions);

    let catData = null;
    try {
      const catRes = await fetch(
        `http://localhost:8080/api/media/csv/categorized_data_${accountId}.csv?t=${timestamp}`,
      );
      if (catRes.ok) {
        const catJson = await catRes.json();
        if (catJson.data && catJson.data.length > 0) catData = catJson.data;
      }
    } catch (e) {
      console.error("Error occured", e);
    }

    let balUrl = "";
    try {
      const balRes = await fetch(
        `http://localhost:8080/api/media/image/monte_carlo_forecast_paths_${accountId}.png?t=${timestamp}`,
        { method: "HEAD" },
      );
      if (balRes.ok)
        balUrl = `http://localhost:8080/api/media/image/monte_carlo_forecast_paths_${accountId}.png?t=${timestamp}`;
    } catch (e) {
      console.error("Error occured", e);
    }

    let expUrl = "";
    try {
      const expRes = await fetch(
        `http://localhost:8080/api/media/image/expense_convergence_path_${accountId}.png?t=${timestamp}`,
        { method: "HEAD" },
      );
      if (expRes.ok)
        expUrl = `http://localhost:8080/api/media/image/expense_convergence_path_${accountId}.png?t=${timestamp}`;
    } catch (e) {
      console.error("Error occured", e);
    }

    setMediaCache({
      categorized: catData,
      balanceForecastUrl: balUrl,
      expenseForecastUrl: expUrl,
    });

    setDisplayMode("account_details");
    setActiveTab("raw");
  };

  const handleAiDisplayTrigger = async (type: TabType) => {
    if (activeAccountId) {
      await fetchAccountDetails(activeAccountId);
      setActiveTab(type);
    } else if (accounts.length > 0) {
      const targetAccountId = accounts[0].account_id;
      await fetchAccountDetails(targetAccountId);
      setActiveTab(type);
    }
  };

  return (
    <div className="h-screen bg-[#0D1117] text-white flex overflow-hidden">
      <aside className="w-[22%] p-6 flex flex-col gap-4 overflow-y-auto shrink-0 scrollbar-hide">
        <h2 className="text-xs font-bold tracking-widest text-slate-500 mb-2">
          LINKED ACCOUNTS
        </h2>
        {accounts.map((acc, idx) => (
          <div
            key={idx}
            onClick={() => fetchAccountDetails(acc.account_id)}
            // Subtle active-state border highlight so the user knows which account is focused
            className={`bg-[#161B22] p-5 rounded-2xl border ${activeAccountId === acc.account_id ? "border-[#00FFAA]" : "border-slate-800 hover:border-[#00FFAA]/50"} cursor-pointer transition-all group shrink-0`}
          >
            <div className="flex items-center gap-3 mb-4">
              <CreditCard className="text-[#00FFAA] w-6 h-6 group-hover:scale-110 transition-transform shrink-0" />
              <div className="flex flex-col truncate">
                <span className="text-sm font-bold text-slate-200 truncate">
                  {acc.provider_name || "Unknown Bank"}
                </span>
                <span className="text-[10px] text-slate-500 font-mono tracking-widest mt-0.5">
                  {acc.sort_code} | ••••{acc.account_number}
                </span>
              </div>
            </div>
            <h3 className="text-2xl font-mono font-bold text-white">
              {acc.currency === "GBP" ? "£" : acc.currency}
              {acc.balance.toLocaleString()}
            </h3>
          </div>
        ))}

        <button
          onClick={async () => {
            const res = await fetch(
              "http://localhost:8080/api/truelayer/status",
            );
            const data = await res.json();
            if (data.auth_url) window.location.href = data.auth_url;
          }}
          className="w-full mt-2 flex items-center justify-center gap-2 bg-[#161B22] border border-slate-800 text-slate-400 p-4 rounded-2xl hover:border-[#00FFAA]/50 hover:text-[#00FFAA] transition-all border-dashed"
        >
          <Plus size={18} />
          <span className="text-xs font-bold tracking-widest uppercase">
            Link Additional Bank
          </span>
        </button>
      </aside>

      <main className="flex-1 flex flex-col relative bg-[#0D1117]">
        <div className="flex-1 p-8 pb-4 overflow-hidden">
          {displayMode === "commodities" && (
            <div className="h-full flex flex-col items-center justify-center text-slate-600 border-2 border-dashed border-slate-800 rounded-3xl">
              <BarChart3 className="w-16 h-16 mb-4 opacity-50" />
              <p className="font-bold tracking-widest uppercase text-sm">
                Commodities & Market Graphs
              </p>
              <p className="text-xs opacity-50">(Future Development Area)</p>
            </div>
          )}

          {displayMode === "account_details" && selectedAccTransactions && (
            <div className="h-full flex flex-col bg-[#161B22] border border-slate-800 rounded-3xl p-6 shadow-2xl animate-in fade-in slide-in-from-bottom-4 overflow-hidden">
              <div className="flex justify-between items-center mb-4 shrink-0">
                <h2 className="text-lg font-bold flex items-center gap-2 text-[#00FFAA]">
                  <History className="w-5 h-5" /> Account Analysis
                </h2>
                <button
                  onClick={() => {
                    setDisplayMode("commodities");
                    setActiveAccountId(null);
                  }}
                  className="flex items-center gap-2 text-xs font-bold text-slate-400 hover:text-white transition-colors bg-slate-800/50 px-3 py-1.5 rounded-lg"
                >
                  <X className="w-4 h-4" /> Close View
                </button>
              </div>

              <div className="flex gap-2 mb-4 border-b border-slate-800 pb-3 shrink-0 overflow-x-auto scrollbar-hide">
                <button
                  onClick={() => setActiveTab("raw")}
                  className={`px-4 py-2 text-xs font-bold rounded-xl transition-all ${activeTab === "raw" ? "bg-[#00FFAA] text-black shadow-lg" : "bg-[#0D1117] text-slate-400 hover:text-white border border-slate-800"}`}
                >
                  Recent Transactions
                </button>
                <button
                  onClick={() => setActiveTab("categorized")}
                  className={`px-4 py-2 text-xs font-bold rounded-xl transition-all ${activeTab === "categorized" ? "bg-[#00FFAA] text-black shadow-lg" : "bg-[#0D1117] text-slate-400 hover:text-white border border-slate-800"}`}
                >
                  Categorized Ledger
                </button>
                <button
                  onClick={() => setActiveTab("balance_forecast")}
                  className={`px-4 py-2 text-xs font-bold rounded-xl transition-all ${activeTab === "balance_forecast" ? "bg-[#00FFAA] text-black shadow-lg" : "bg-[#0D1117] text-slate-400 hover:text-white border border-slate-800"}`}
                >
                  Balance Forecast
                </button>
                <button
                  onClick={() => setActiveTab("expense_forecast")}
                  className={`px-4 py-2 text-xs font-bold rounded-xl transition-all ${activeTab === "expense_forecast" ? "bg-[#00FFAA] text-black shadow-lg" : "bg-[#0D1117] text-slate-400 hover:text-white border border-slate-800"}`}
                >
                  Expense Forecast
                </button>
              </div>

              <div className="flex-1 overflow-y-auto pr-2 scrollbar-hide relative">
                {activeTab === "raw" && (
                  <TransactionFeed
                    transactions={selectedAccTransactions}
                    showCategory={false}
                  />
                )}
                {activeTab === "categorized" &&
                  (mediaCache.categorized ? (
                    <TransactionFeed
                      transactions={mediaCache.categorized}
                      showCategory={true}
                    />
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-4">
                      <ShieldCheck className="w-12 h-12 mb-4 text-slate-700" />
                      <p className="text-sm text-slate-400 font-medium">
                        Categorize your transactions between a given range with
                        BudAI first.
                      </p>
                    </div>
                  ))}
                {activeTab === "balance_forecast" &&
                  (mediaCache.balanceForecastUrl ? (
                    <div className="w-full h-full relative flex items-center justify-center">
                      <Image
                        src={mediaCache.balanceForecastUrl}
                        alt="Balance Forecast"
                        fill
                        unoptimized
                        className="object-contain rounded-xl border border-slate-800"
                      />
                    </div>
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-4">
                      <TrendingUp className="w-12 h-12 mb-4 text-slate-700" />
                      <p className="text-sm text-slate-400 font-medium">
                        Run a balance forecast with BudAI first.
                      </p>
                    </div>
                  ))}
                {activeTab === "expense_forecast" &&
                  (mediaCache.expenseForecastUrl ? (
                    <div className="w-full h-full relative flex items-center justify-center">
                      <Image
                        src={mediaCache.expenseForecastUrl}
                        alt="Expense Forecast"
                        fill
                        unoptimized
                        className="object-contain rounded-xl border border-slate-800"
                      />
                    </div>
                  ) : (
                    <div className="absolute inset-0 flex flex-col items-center justify-center text-center px-4">
                      <TrendingDown className="w-12 h-12 mb-4 text-slate-700" />
                      <p className="text-sm text-slate-400 font-medium">
                        Run an expense forecast with BudAI first.
                      </p>
                    </div>
                  ))}
              </div>
            </div>
          )}
        </div>

        <div className="w-full h-[45vh] shrink-0 px-8 pb-8">
          <BudAIChat
            onAiAction={handleAiDisplayTrigger}
            activeAccountId={activeAccountId}
          />
        </div>
      </main>

      <aside className="w-[25%] p-6 bg-[#161B22] border-l border-slate-800 overflow-y-auto shrink-0 scrollbar-hide">
        <h2 className="text-xs font-bold tracking-widest text-slate-500 mb-6 flex items-center gap-2">
          <Globe className="w-4 h-4" /> GLOBAL MARKETS (DEV)
        </h2>
        <div className="space-y-4">
          {[1, 2, 3].map((item) => (
            <div
              key={item}
              className="bg-[#0D1117] p-5 rounded-2xl border border-slate-800"
            >
              <div className="h-4 w-1/4 bg-slate-800 rounded mb-3"></div>
              <div className="h-4 w-full bg-slate-800 rounded mb-2"></div>
              <div className="h-4 w-5/6 bg-slate-800 rounded"></div>
              <p className="text-[10px] text-slate-500 mt-4 uppercase">
                Geopolitical Placeholder
              </p>
            </div>
          ))}
        </div>
      </aside>
    </div>
  );
}
