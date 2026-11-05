"use client";

import React, { useState, useMemo, useEffect, useCallback } from "react";
import { Button, Skeleton, Dropdown, Calendar } from "@heroui/react";
import {
  TrendingUp,
  ShieldCheck,
  Calendar as CalendarIcon,
  Sparkles,
  ChevronDown,
  ChevronLeft,
  ChevronRight,
  ArrowLeft,
  ArrowRight,
  PieChart,
  BarChart2,
  Clock,
  Target,
  ShoppingBag,
  CreditCard,
  DollarSign,
  Layers,
  ArrowUpRight,
  ArrowDownRight,
  CheckCircle2,
} from "lucide-react";
import { ChartConfiguration } from "chart.js";
import CoreChartEngine from "@/app/(protected)/_components/internal/ChartEngine";
import { useBudAI } from "@/app/context/AppContext";
import {
  useSpendingTrends,
  useTransactions,
  useExpenseCategories,
  useBuckets,
  useSubscriptions,
  useUserProfile,
} from "@/lib/hooks";
import { formatCurrency, getCurrencySymbol } from "@/lib/currency";
import { apiFetch } from "@/lib/api";
import { today, getLocalTimeZone, CalendarDate } from "@internationalized/date";

export type CardViewMode =
  | "forecast"
  | "calendar"
  | "expenses"
  | "cashflow"
  | "pace"
  | "subscriptions"
  | "goals"
  | "wants_needs"
  | "merchants"
  | "allowance";

export interface FinancialAnalysisCardProps {
  currency?: string;
}

export const FinancialAnalysisCard = ({
  currency = "GBP",
}: FinancialAnalysisCardProps) => {
  const { accounts, activeAccountId } = useBudAI();

  // Active View State
  const [activeView, setActiveView] = useState<CardViewMode>("forecast");

  // Trend & Forecast Horizons (Symmetrical Past & Future: 30d, 60d, 90d)
  const [pastLookbackDays, setPastLookbackDays] = useState<30 | 60 | 90>(30);
  const [horizonDays, setHorizonDays] = useState<30 | 60 | 90>(30);
  const [isForecastExtended, setIsForecastExtended] = useState<boolean>(false);
  const [isFetchingBackendForecast, setIsFetchingBackendForecast] = useState<boolean>(false);
  const [backendForecastSeries, setBackendForecastSeries] = useState<any[] | null>(null);
  const [safetyCushion, setSafetyCushion] = useState<number>(500);

  // Calendar State (Spanning -50 Years to +5 Years)
  const todayDate = useMemo(() => today(getLocalTimeZone()), []);
  const minCalendarDate = useMemo(() => todayDate.subtract({ years: 50 }), [todayDate]);
  const maxCalendarDate = useMemo(() => todayDate.add({ years: 5 }), [todayDate]);
  const [selectedCalendarDate, setSelectedCalendarDate] = useState<CalendarDate>(todayDate);
  const [focusedValue, setFocusedValue] = useState<CalendarDate>(todayDate);

  const handleToday = () => {
    setFocusedValue(todayDate);
    setSelectedCalendarDate(todayDate);
  };

  const handleDateCellClick = (date: CalendarDate) => {
    if (date.month !== focusedValue.month) {
      setFocusedValue(date);
    }
    setSelectedCalendarDate(date);
  };

  useEffect(() => {
    if (typeof window !== "undefined") {
      const saved = localStorage.getItem("budai_safety_cushion");
      if (saved) {
        const val = parseFloat(saved);
        if (!isNaN(val)) setSafetyCushion(val);
      }
    }
  }, []);

  const activeAccount = useMemo(() => {
    if (!activeAccountId || activeAccountId === "ALL") return accounts[0] || null;
    return accounts.find((a) => a.account_id === activeAccountId) || accounts[0] || null;
  }, [accounts, activeAccountId]);

  const activeCurrency = activeAccount?.currency || currency;
  const currentBalance = activeAccount?.balance ?? 0;
  const targetAccountId = activeAccount?.account_id || "";

  // Date range for past lookback (30d, 60d, 90d)
  const { fromDate, toDate } = useMemo(() => {
    const end = new Date();
    const start = new Date();
    start.setDate(end.getDate() - pastLookbackDays);
    return {
      fromDate: start.toISOString().split("T")[0],
      toDate: end.toISOString().split("T")[0],
    };
  }, [pastLookbackDays]);

  // Longer window for transactions and cashflow analysis (last 180 days)
  const { sixMonthsAgoStr, todayStr } = useMemo(() => {
    const end = new Date();
    const start = new Date();
    start.setDate(end.getDate() - 180);
    return {
      sixMonthsAgoStr: start.toISOString().split("T")[0],
      todayStr: end.toISOString().split("T")[0],
    };
  }, []);

  const { data: trendData = [], isLoading: isTrendLoading } = useSpendingTrends(
    targetAccountId,
    fromDate,
    toDate,
    "daily",
    undefined,
    !!targetAccountId,
  );

  const { data: transactions = [], isLoading: isTxLoading } = useTransactions(
    targetAccountId,
    sixMonthsAgoStr,
    todayStr,
  );

  const { data: expenseCatData = [], isLoading: isCatLoading } = useExpenseCategories(
    targetAccountId,
    fromDate,
    toDate,
    !!targetAccountId,
  );

  const { data: userProfile } = useUserProfile();
  const { data: buckets = [] } = useBuckets();
  const { data: detectedSubs = [] } = useSubscriptions(userProfile?.user_uuid);

  // Backend forecast trigger
  const triggerBackendForecast = useCallback(
    async (days: number) => {
      if (!targetAccountId) return;
      setIsFetchingBackendForecast(true);
      try {
        // Direct deterministic balance forecast endpoint
        const directRes = await apiFetch(
          `/api/widgets/forecast/balance?account_id=${encodeURIComponent(targetAccountId)}&days=${days}`,
          { method: "GET" },
          true,
        );

        if (directRes.ok) {
          const directJson = (await directRes.json()) as any;
          if (directJson?.data?.series?.[0]?.data) {
            setBackendForecastSeries(directJson.data.series[0].data);
            return;
          }
        }
      } catch (err) {
        console.error("Balance forecast calculation error:", err);
      } finally {
        setIsFetchingBackendForecast(false);
      }
    },
    [targetAccountId],
  );

  const handleToggleForecast = () => {
    const nextState = !isForecastExtended;
    setIsForecastExtended(nextState);
    if (nextState && !backendForecastSeries) {
      triggerBackendForecast(horizonDays);
    }
  };

  const handleHorizonChange = (days: 30 | 60 | 90) => {
    setHorizonDays(days);
    if (isForecastExtended) {
      triggerBackendForecast(days);
    }
  };

  // ---------------------------------------------------------------------------
  // 1. DATA COMPUTATION FOR ALL VIEWS
  // ---------------------------------------------------------------------------

  // A. Forecast & Past Daily Balances
  const { labels, pastBalances, avgDailySpend, projectedBalance } = useMemo(() => {
    const dailyPoints =
      trendData && trendData.length > 0 && trendData[0]?.data
        ? trendData[0].data
        : [];

    const totalDays = pastLookbackDays;
    const computedLabels: string[] = [];
    const balances: number[] = [];
    let runningBalance = currentBalance;

    let totalRecentSpend = 0;
    dailyPoints.forEach((p: any) => {
      const val = parseFloat(p.Amount || p.amount || 0);
      if (val > 0) totalRecentSpend += val;
    });

    const calculatedDailyBurn =
      dailyPoints.length > 0
        ? totalRecentSpend / dailyPoints.length
        : currentBalance > 0
        ? Math.max(15, currentBalance * 0.015)
        : 25;

    for (let i = totalDays - 1; i >= 0; i--) {
      const d = new Date();
      d.setDate(d.getDate() - i);
      computedLabels.push(
        d.toLocaleDateString("en-GB", { month: "short", day: "numeric" }),
      );

      const matchingPoint = dailyPoints.find((p: any) => {
        const ptDate = p.Date || p.date || "";
        return ptDate.endsWith(
          `-${String(d.getMonth() + 1).padStart(2, "0")}-${String(
            d.getDate(),
          ).padStart(2, "0")}`,
        );
      });

      const daySpend = matchingPoint
        ? parseFloat(matchingPoint.Amount || matchingPoint.amount || 0)
        : calculatedDailyBurn;

      balances.push(Math.round(runningBalance));
      runningBalance = Math.max(0, runningBalance - daySpend);
    }

    const projected = Math.max(
      0,
      currentBalance - calculatedDailyBurn * horizonDays,
    );

    return {
      labels: computedLabels,
      pastBalances: balances,
      avgDailySpend: Math.round(calculatedDailyBurn),
      projectedBalance: Math.round(projected),
    };
  }, [trendData, pastLookbackDays, currentBalance, horizonDays]);

  const forecastBalances = useMemo(() => {
    if (backendForecastSeries && backendForecastSeries.length > 0) {
      return backendForecastSeries.slice(0, horizonDays).map((p: any) => {
        if (typeof p === "number") return Math.round(p);
        const val =
          p["Expected Balance"] ??
          p["expected_balance"] ??
          p.balance ??
          p.y ??
          p.Amount ??
          0;
        return Math.round(val);
      });
    }
    const series = [];
    let cur = currentBalance;
    for (let i = 1; i <= horizonDays; i++) {
      cur = Math.max(0, cur - avgDailySpend);
      series.push(Math.round(cur));
    }
    return series;
  }, [backendForecastSeries, currentBalance, avgDailySpend, horizonDays]);

  // B. Calendar Events (Bills, Subscriptions, Paydays)
  const calendarEvents = useMemo(() => {
    const events: Array<{
      id: string;
      title: string;
      amount: number;
      dateStr: string;
      dayOfMonth: number;
      type: "bill" | "subscription" | "income";
      status: string;
    }> = [];

    // 1. Detected Subscriptions from analytics pipeline
    if (Array.isArray(detectedSubs)) {
      detectedSubs.forEach((s: any, idx: number) => {
        const dateStr = s.next_expected_date ? s.next_expected_date.split("T")[0] : "";
        const day = dateStr ? parseInt(dateStr.split("-")[2], 10) : 1;
        const amt = Number(s.expected_amount || s.last_payment_amount || 0);
        events.push({
          id: `sub-${idx}-${s.merchant_name || "sub"}`,
          title: s.merchant_name || "Subscription",
          amount: Math.abs(amt),
          dateStr,
          dayOfMonth: isNaN(day) ? 1 : day,
          type: "subscription",
          status: s.status || (s.is_price_hike ? "Price Increase Flagged" : "Active"),
        });
      });
    }

    // 2. Virtual Buckets (Recurring commitments & target dates)
    if (Array.isArray(buckets)) {
      buckets.forEach((b: any, idx: number) => {
        if (b.type === "RECURRING" || b.type === "TARGET_DATE" || b.type === "LIABILITY") {
          const dateStr = b.target_date ? b.target_date.split("T")[0] : "";
          const day = dateStr ? parseInt(dateStr.split("-")[2], 10) : 1;
          const amt = Number(b.target_amount || b.cached_balance || 0);
          const alreadyAdded = events.some(
            (e) => e.title.toLowerCase() === (b.name || "").toLowerCase(),
          );
          if (!alreadyAdded) {
            events.push({
              id: `bucket-${b.id || idx}`,
              title: b.name || "Obligation",
              amount: Math.abs(amt),
              dateStr,
              dayOfMonth: isNaN(day) ? 1 : day,
              type: "bill",
              status: "Virtual Reserve",
            });
          }
        }
      });
    }

    // 3. Transactions (Settled historical entries)
    if (Array.isArray(transactions)) {
      transactions.forEach((tx) => {
        const txDate = tx.timestamp ? tx.timestamp.split("T")[0] : "";
        if (!txDate) return;
        const day = parseInt(txDate.split("-")[2], 10) || 1;
        const desc = (tx.description || "").toLowerCase();
        const cat = (tx.category || "").toLowerCase();
        const amount = Math.abs(Number(tx.amount) || 0);

        const isSub =
          cat.includes("subscription") ||
          desc.includes("netflix") ||
          desc.includes("spotify") ||
          desc.includes("prime") ||
          desc.includes("apple") ||
          desc.includes("gym");

        const isBill =
          cat.includes("utilities") ||
          cat.includes("bill") ||
          cat.includes("housing") ||
          cat.includes("rent") ||
          cat.includes("council") ||
          desc.includes("water") ||
          desc.includes("gas") ||
          desc.includes("energy");

        const isInc = (Number(tx.amount) || 0) > 0 || cat.includes("income") || desc.includes("salary");

        if (isSub) {
          events.push({
            id: tx.transaction_id || Math.random().toString(),
            title: tx.description || "Subscription",
            amount,
            dateStr: txDate,
            dayOfMonth: day,
            type: "subscription",
            status: "Settled",
          });
        } else if (isBill) {
          events.push({
            id: tx.transaction_id || Math.random().toString(),
            title: tx.description || "Utility Bill",
            amount,
            dateStr: txDate,
            dayOfMonth: day,
            type: "bill",
            status: "Settled",
          });
        } else if (isInc) {
          events.push({
            id: tx.transaction_id || Math.random().toString(),
            title: tx.description || "Income Deposit",
            amount,
            dateStr: txDate,
            dayOfMonth: day,
            type: "income",
            status: "Settled",
          });
        }
      });
    }

    return events;
  }, [detectedSubs, buckets, transactions]);

  // Selected Date events in calendar
  const selectedDateEvents = useMemo(() => {
    const formatted = `${selectedCalendarDate.year}-${String(
      selectedCalendarDate.month,
    ).padStart(2, "0")}-${String(selectedCalendarDate.day).padStart(2, "0")}`;

    // Match exact date or recurring day of month
    return calendarEvents.filter(
      (e) => e.dateStr === formatted || e.dayOfMonth === selectedCalendarDate.day,
    );
  }, [calendarEvents, selectedCalendarDate]);

  // Total bills and subscriptions this month
  const { totalBillsMonth, totalSubsMonth } = useMemo(() => {
    let bills = 0;
    let subs = 0;

    if (Array.isArray(detectedSubs) && detectedSubs.length > 0) {
      detectedSubs.forEach((s: any) => {
        subs += Number(s.expected_amount || s.last_payment_amount || 0);
      });
    }

    if (Array.isArray(buckets)) {
      buckets.forEach((b: any) => {
        if (b.type === "RECURRING" || b.type === "LIABILITY") {
          bills += Number(b.target_amount || b.cached_balance || 0);
        }
      });
    }

    if (bills === 0 || subs === 0) {
      calendarEvents.forEach((e) => {
        if (e.type === "bill" && bills === 0) bills += e.amount;
        if (e.type === "subscription" && subs === 0) subs += e.amount;
      });
    }

    return {
      totalBillsMonth: Math.round(bills * 100) / 100,
      totalSubsMonth: Math.round(subs * 100) / 100,
    };
  }, [detectedSubs, buckets, calendarEvents]);

  // C. Expense Breakdown Data (Categories Donut / Pie)
  const categoryData = useMemo(() => {
    if (expenseCatData && expenseCatData.length > 0 && expenseCatData[0]?.data && expenseCatData[0].data.length > 0) {
      return expenseCatData[0].data
        .map((item: any) => ({
          name: item.Category || item.category || "General Spending",
          value: parseFloat(item.Total_Amount || item.total_amount || item.Amount || item.amount || 0),
        }))
        .filter((i: any) => i.value > 0)
        .sort((a: any, b: any) => b.value - a.value);
    }

    // Dynamic aggregation from transactions
    const catMap: Record<string, number> = {};
    if (Array.isArray(transactions)) {
      transactions.forEach((tx) => {
        const amt = Number(tx.amount) || 0;
        if (amt < 0) {
          const cat = tx.category || "General Spending";
          catMap[cat] = (catMap[cat] || 0) + Math.abs(amt);
        }
      });
    }

    return Object.entries(catMap)
      .map(([name, value]) => ({
        name,
        value: Math.round(value * 100) / 100,
      }))
      .sort((a, b) => b.value - a.value);
  }, [expenseCatData, transactions]);

  const totalExpenseAmount = useMemo(() => {
    return categoryData.reduce((acc, curr) => acc + curr.value, 0);
  }, [categoryData]);

  // D. Cash Flow (Monthly In vs Out for last 6 calendar months)
  const cashflowMonths = useMemo(() => {
    const monthKeys: string[] = [];
    const monthLabels: string[] = [];
    const inByMonth: Record<string, number> = {};
    const outByMonth: Record<string, number> = {};

    for (let i = 5; i >= 0; i--) {
      const d = new Date();
      d.setDate(1);
      d.setMonth(d.getMonth() - i);
      const key = `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, "0")}`;
      monthKeys.push(key);
      monthLabels.push(d.toLocaleDateString("en-GB", { month: "short" }));
      inByMonth[key] = 0;
      outByMonth[key] = 0;
    }

    if (Array.isArray(transactions)) {
      transactions.forEach((tx) => {
        const txDate = tx.timestamp ? tx.timestamp.split("T")[0] : "";
        if (!txDate) return;
        const key = txDate.substring(0, 7);
        if (key in inByMonth) {
          const amt = Number(tx.amount) || 0;
          if (amt > 0) {
            inByMonth[key] += amt;
          } else if (amt < 0) {
            outByMonth[key] += Math.abs(amt);
          }
        }
      });
    }

    const inflow = monthKeys.map((k) => Math.round(inByMonth[k]));
    const outflow = monthKeys.map((k) => Math.round(outByMonth[k]));

    const currentKey = monthKeys[monthKeys.length - 1];
    const prevKey = monthKeys[monthKeys.length - 2];

    const thisMonthIn = inByMonth[currentKey] || 0;
    const thisMonthOut = outByMonth[currentKey] || 0;
    const thisMonthNet = thisMonthIn - thisMonthOut;

    const prevMonthOut = outByMonth[prevKey] || 0;
    const spendDeltaPct =
      prevMonthOut > 0
        ? Math.round(((thisMonthOut - prevMonthOut) / prevMonthOut) * 100)
        : 0;

    return {
      labels: monthLabels,
      inflow,
      outflow,
      thisMonthIn,
      thisMonthOut,
      thisMonthNet,
      spendDeltaPct,
    };
  }, [transactions]);

  // E. Subscriptions List (View 6)
  const subscriptionsList = useMemo(() => {
    const list: Array<{
      name: string;
      cost: number;
      renewal: string;
      isHike?: boolean;
    }> = [];

    if (Array.isArray(detectedSubs)) {
      detectedSubs.forEach((s: any) => {
        let renewal = "Active";
        if (s.next_expected_date) {
          const nextD = new Date(s.next_expected_date);
          const diffMs = nextD.getTime() - Date.now();
          const diffDays = Math.ceil(diffMs / (1000 * 60 * 60 * 24));
          if (diffDays > 0) {
            renewal = `Due in ${diffDays} day${diffDays === 1 ? "" : "s"}`;
          } else if (diffDays === 0) {
            renewal = "Due today";
          } else {
            renewal = `Billed on ${nextD.toLocaleDateString("en-GB", { day: "numeric", month: "short" })}`;
          }
        }
        list.push({
          name: s.merchant_name || "Subscription",
          cost: Number(s.expected_amount || s.last_payment_amount || 0),
          renewal,
          isHike: !!s.is_price_hike,
        });
      });
    }

    if (Array.isArray(buckets)) {
      buckets.forEach((b: any) => {
        if (b.type === "RECURRING") {
          const exists = list.some(
            (sub) => sub.name.toLowerCase() === (b.name || "").toLowerCase(),
          );
          if (!exists) {
            list.push({
              name: b.name || "Recurring Outflow",
              cost: Number(b.target_amount || b.cached_balance || 0),
              renewal: "Virtual Reserve",
            });
          }
        }
      });
    }

    return list;
  }, [detectedSubs, buckets]);

  // F. Savings Goals (View 7)
  const savingsGoals = useMemo(() => {
    if (!Array.isArray(buckets) || buckets.length === 0) return [];
    return buckets
      .filter(
        (b: any) =>
          b.type === "ACCUMULATING" ||
          b.type === "TARGET_DATE" ||
          b.type === "DEFAULT",
      )
      .map((b: any) => {
        const current = Number(b.cached_balance) || 0;
        const target = Number(b.target_amount) || current || 1;
        const targetDateStr = b.target_date
          ? new Date(b.target_date).toLocaleDateString("en-GB", {
              month: "short",
              year: "numeric",
            })
          : "Ongoing";
        const pct = target > 0 ? Math.min(100, Math.round((current / target) * 100)) : 100;
        return {
          name: b.name || "Capital Bucket",
          current,
          target,
          date: targetDateStr,
          pct,
        };
      });
  }, [buckets]);

  // G. Wants vs Needs 50/30/20 (View 8)
  const wantsNeedsData = useMemo(() => {
    let needsAmt = 0;
    let wantsAmt = 0;

    const needsKeywords = [
      "housing",
      "rent",
      "utilities",
      "grocery",
      "groceries",
      "supermarket",
      "bill",
      "bills",
      "transport",
      "transit",
      "council",
      "health",
      "medical",
      "insurance",
      "energy",
      "gas",
      "water",
    ];

    if (Array.isArray(transactions)) {
      transactions.forEach((tx) => {
        const amt = Number(tx.amount) || 0;
        if (amt < 0) {
          const cat = (tx.category || "").toLowerCase();
          const desc = (tx.description || "").toLowerCase();
          const isNeed = needsKeywords.some(
            (k) => cat.includes(k) || desc.includes(k),
          );
          if (isNeed) {
            needsAmt += Math.abs(amt);
          } else {
            wantsAmt += Math.abs(amt);
          }
        }
      });
    }

    const netSavings = Math.max(0, cashflowMonths.thisMonthNet);
    const totalAllocated = needsAmt + wantsAmt + netSavings;

    const needsPct = totalAllocated > 0 ? Math.round((needsAmt / totalAllocated) * 100) : 0;
    const wantsPct = totalAllocated > 0 ? Math.round((wantsAmt / totalAllocated) * 100) : 0;
    const savingsPct = totalAllocated > 0 ? Math.max(0, 100 - needsPct - wantsPct) : 0;

    return [
      {
        label: "Essential Needs (Target ~50%)",
        pct: needsPct,
        amount: Math.round(needsAmt),
        color: "bg-blue-500",
      },
      {
        label: "Flexible Wants (Target ~30%)",
        pct: wantsPct,
        amount: Math.round(wantsAmt),
        color: "bg-purple-500",
      },
      {
        label: "Retained Savings (Target ~20%)",
        pct: savingsPct,
        amount: Math.round(netSavings),
        color: "bg-emerald-500",
      },
    ];
  }, [transactions, cashflowMonths.thisMonthNet]);

  // H. Top Merchants (View 9)
  const topMerchants = useMemo(() => {
    const map: Record<string, { name: string; amount: number; count: number }> = {};
    if (Array.isArray(transactions)) {
      transactions.forEach((tx) => {
        const amt = Number(tx.amount) || 0;
        if (amt < 0) {
          const name = tx.description || tx.category || "General Outflow";
          if (!map[name]) {
            map[name] = { name, amount: 0, count: 0 };
          }
          map[name].amount += Math.abs(amt);
          map[name].count += 1;
        }
      });
    }

    return Object.values(map)
      .sort((a, b) => b.amount - a.amount)
      .slice(0, 6)
      .map((item, idx) => ({
        rank: idx + 1,
        name: item.name,
        amount: Math.round(item.amount * 100) / 100,
        count: `${item.count} Transaction${item.count === 1 ? "" : "s"}`,
      }));
  }, [transactions]);

  // I. Daily Allowance & Payday Forecast (View 10)
  const allowanceInfo = useMemo(() => {
    const incomeDates: number[] = [];
    if (Array.isArray(transactions)) {
      transactions.forEach((tx) => {
        const amt = Number(tx.amount) || 0;
        const desc = (tx.description || "").toLowerCase();
        const cat = (tx.category || "").toLowerCase();
        if ((amt >= 500 || cat.includes("income") || desc.includes("salary")) && tx.timestamp) {
          const day = parseInt(tx.timestamp.split("T")[0].split("-")[2], 10);
          if (day) incomeDates.push(day);
        }
      });
    }

    let typicalPayDay = 28;
    if (incomeDates.length > 0) {
      const frequency: Record<number, number> = {};
      incomeDates.forEach((d) => {
        frequency[d] = (frequency[d] || 0) + 1;
      });
      typicalPayDay = parseInt(
        Object.keys(frequency).reduce((a, b) =>
          frequency[parseInt(a, 10)] > frequency[parseInt(b, 10)] ? a : b,
        ),
        10,
      );
    }

    const now = new Date();
    let nextPayDate = new Date(now.getFullYear(), now.getMonth(), typicalPayDay);
    if (nextPayDate.getTime() <= now.getTime()) {
      nextPayDate = new Date(now.getFullYear(), now.getMonth() + 1, typicalPayDay);
    }

    const diffMs = nextPayDate.getTime() - now.getTime();
    const daysRemaining = Math.max(1, Math.ceil(diffMs / (1000 * 60 * 60 * 24)));
    const safeDailySpend = Math.max(0, (currentBalance - safetyCushion) / daysRemaining);

    return {
      safeDailySpend: Math.round(safeDailySpend * 100) / 100,
      daysRemaining,
      expectedDateStr: nextPayDate.toLocaleDateString("en-GB", {
        day: "numeric",
        month: "short",
        year: "numeric",
      }),
    };
  }, [transactions, currentBalance, safetyCushion]);

  // ---------------------------------------------------------------------------
  // 2. CHART CONFIGURATIONS
  // ---------------------------------------------------------------------------

  // Chart 1: Balance Trend & Forecast Line Chart
  const forecastChartConfig: ChartConfiguration = useMemo(() => {
    const combinedLabels = [...labels];
    if (isForecastExtended) {
      for (let i = 1; i <= horizonDays; i++) {
        const d = new Date();
        d.setDate(d.getDate() + i);
        combinedLabels.push(
          d.toLocaleDateString("en-GB", { month: "short", day: "numeric" }),
        );
      }
    }

    const pastSeriesPadded = [
      ...pastBalances,
      ...(isForecastExtended ? new Array(horizonDays).fill(null) : []),
    ];

    const forecastSeriesPadded = isForecastExtended
      ? [
          ...new Array(pastBalances.length - 1).fill(null),
          pastBalances[pastBalances.length - 1] ?? currentBalance,
          ...forecastBalances,
        ]
      : [];

    return {
      type: "line",
      data: {
        labels: combinedLabels,
        datasets: [
          {
            label: `Past Balance (${pastLookbackDays}d)`,
            data: pastSeriesPadded as any,
            borderColor: "rgba(59, 130, 246, 1)",
            backgroundColor: "rgba(59, 130, 246, 0.08)",
            fill: true,
            tension: 0.35,
            pointRadius: 0,
            pointHoverRadius: 4,
            borderWidth: 2,
          },
          ...(isForecastExtended
            ? [
                {
                  label: `Forecast Trajectory (${horizonDays}d)`,
                  data: forecastSeriesPadded as any,
                  borderColor: "rgba(168, 85, 247, 1)",
                  backgroundColor: "rgba(168, 85, 247, 0.05)",
                  borderDash: [5, 5],
                  fill: true,
                  tension: 0.35,
                  pointRadius: 0,
                  pointHoverRadius: 4,
                  borderWidth: 2,
                },
              ]
            : []),
          {
            label: "Safety Floor Cushion",
            data: new Array(combinedLabels.length).fill(safetyCushion),
            borderColor: "rgba(239, 68, 68, 0.7)",
            borderDash: [4, 4],
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                const val = context.parsed.y ?? 0;
                return `${context.dataset.label}: ${getCurrencySymbol(
                  activeCurrency,
                )}${val.toLocaleString(undefined, {
                  minimumFractionDigits: 0,
                  maximumFractionDigits: 0,
                })}`;
              },
            },
          },
        },
      },
    };
  }, [
    labels,
    pastBalances,
    forecastBalances,
    isForecastExtended,
    horizonDays,
    pastLookbackDays,
    safetyCushion,
    activeCurrency,
    currentBalance,
  ]);

  // Chart 2: Expenses Donut / Pie Chart
  const pieColors = [
    "#3b82f6",
    "#10b981",
    "#a855f7",
    "#f59e0b",
    "#ec4899",
    "#06b6d4",
    "#ef4444",
    "#6366f1",
  ];

  const expensesPieConfig: ChartConfiguration = useMemo(() => {
    const topCategories = categoryData.slice(0, 7);
    return {
      type: "doughnut",
      data: {
        labels: topCategories.map((c) => c.name),
        datasets: [
          {
            data: topCategories.map((c) => c.value),
            backgroundColor: pieColors.slice(0, topCategories.length),
            borderColor: "rgba(13, 14, 20, 0.8)",
            borderWidth: 2,
            hoverOffset: 4,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        cutout: "70%",
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (context) => {
                const val = (context.raw as number) ?? 0;
                const pct =
                  totalExpenseAmount > 0
                    ? Math.round((val / totalExpenseAmount) * 100)
                    : 0;
                return ` ${context.label}: ${formatCurrency(
                  val,
                  activeCurrency,
                )} (${pct}%)`;
              },
            },
          },
        },
      },
    };
  }, [categoryData, totalExpenseAmount, activeCurrency]);

  // Chart 3: Cash Flow Bar Chart
  const cashflowBarConfig: ChartConfiguration = useMemo(() => {
    return {
      type: "bar",
      data: {
        labels: cashflowMonths.labels,
        datasets: [
          {
            label: "Money In",
            data: cashflowMonths.inflow,
            backgroundColor: "rgba(16, 185, 129, 0.85)",
            borderColor: "rgba(16, 185, 129, 1)",
            borderWidth: 1,
            borderRadius: 6,
          },
          {
            label: "Money Out",
            data: cashflowMonths.outflow,
            backgroundColor: "rgba(239, 68, 68, 0.75)",
            borderColor: "rgba(239, 68, 68, 1)",
            borderWidth: 1,
            borderRadius: 6,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) =>
                ` ${ctx.dataset.label}: ${formatCurrency(
                  ctx.parsed.y ?? 0,
                  activeCurrency,
                )}`,
            },
          },
        },
      },
    };
  }, [cashflowMonths, activeCurrency]);

  // Chart 4: Spending Pace Line Chart
  const paceChartConfig: ChartConfiguration = useMemo(() => {
    const days = Array.from({ length: 30 }, (_, i) => `Day ${i + 1}`);
    const todayDayOfMonth = new Date().getDate();

    const now = new Date();
    const currentYearMonth = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;
    const prevDate = new Date(now.getFullYear(), now.getMonth() - 1, 1);
    const prevYearMonth = `${prevDate.getFullYear()}-${String(prevDate.getMonth() + 1).padStart(2, "0")}`;

    const thisMonthDaily: number[] = new Array(31).fill(0);
    const lastMonthDaily: number[] = new Array(31).fill(0);

    if (Array.isArray(transactions)) {
      transactions.forEach((tx) => {
        const amt = Number(tx.amount) || 0;
        if (amt < 0 && tx.timestamp) {
          const dateStr = tx.timestamp.split("T")[0];
          const ym = dateStr.substring(0, 7);
          const day = parseInt(dateStr.split("-")[2], 10);
          if (day >= 1 && day <= 31) {
            if (ym === currentYearMonth) {
              thisMonthDaily[day - 1] += Math.abs(amt);
            } else if (ym === prevYearMonth) {
              lastMonthDaily[day - 1] += Math.abs(amt);
            }
          }
        }
      });
    }

    let cumThis = 0;
    let cumLast = 0;
    const thisMonthCurve: (number | null)[] = [];
    const lastMonthCurve: number[] = [];

    for (let i = 0; i < 30; i++) {
      cumLast += lastMonthDaily[i] || 0;
      lastMonthCurve.push(Math.round(cumLast));

      if (i + 1 <= todayDayOfMonth) {
        cumThis += thisMonthDaily[i] || 0;
        thisMonthCurve.push(Math.round(cumThis));
      } else {
        thisMonthCurve.push(null);
      }
    }

    return {
      type: "line",
      data: {
        labels: days,
        datasets: [
          {
            label: "This Month Spend",
            data: thisMonthCurve as any,
            borderColor: "rgba(59, 130, 246, 1)",
            backgroundColor: "rgba(59, 130, 246, 0.08)",
            fill: true,
            tension: 0.3,
            borderWidth: 2,
            pointRadius: 0,
          },
          {
            label: "Last Month Spend",
            data: lastMonthCurve as any,
            borderColor: "rgba(156, 163, 175, 0.6)",
            borderDash: [4, 4],
            tension: 0.3,
            borderWidth: 1.5,
            pointRadius: 0,
            fill: false,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              label: (ctx) =>
                ` ${ctx.dataset.label}: ${formatCurrency(
                  ctx.parsed.y ?? 0,
                  activeCurrency,
                )}`,
            },
          },
        },
      },
    };
  }, [transactions, activeCurrency]);

  // ---------------------------------------------------------------------------
  // 3. TITLE & SUBTITLE PER ACTIVE VIEW
  // ---------------------------------------------------------------------------
  const viewMeta = useMemo(() => {
    switch (activeView) {
      case "forecast":
        return {
          title: "Balance Trend & Forecast",
          subtitle: `Past ${pastLookbackDays}d trend and forward projection`,
          badge: `${pastLookbackDays}-Day Base`,
        };
      case "calendar":
        return {
          title: "Payments & Bills Calendar",
          subtitle: "Scheduled recurring bills and historical payments",
          badge: "Full History",
        };
      case "expenses":
        return {
          title: "Expense Breakdown",
          subtitle: "Outflows grouped by category for this period",
          badge: "Categories",
        };
      case "cashflow":
        return {
          title: "Monthly Cash Flow",
          subtitle: "Income vs. spending and net monthly savings",
          badge: "6-Month Trend",
        };
      case "pace":
        return {
          title: "Spending Pace",
          subtitle: "Day-by-day spending pace compared to last month",
          badge: "Rate",
        };
      case "subscriptions":
        return {
          title: "Active Subscriptions",
          subtitle: "Detected regular subscriptions and memberships",
          badge: "Recurring",
        };
      case "goals":
        return {
          title: "Savings Goals Progress",
          subtitle: "Target reserves and contribution schedules",
          badge: "Allocations",
        };
      case "wants_needs":
        return {
          title: "Needs vs. Wants",
          subtitle: "Essential living vs. flexible lifestyle spending",
          badge: "50/30/20",
        };
      case "merchants":
        return {
          title: "Top Merchants",
          subtitle: "Highest outflow stores and companies this month",
          badge: "Merchants",
        };
      case "allowance":
        return {
          title: "Daily Spending Allowance",
          subtitle: "Safe daily spending limit before next paycheck",
          badge: "Payday Track",
        };
    }
  }, [activeView, pastLookbackDays]);

  const isBelowCushion = projectedBalance < safetyCushion;

  // ---------------------------------------------------------------------------
  // 4. RENDER
  // ---------------------------------------------------------------------------
  return (
    <div className="flex flex-col gap-4 p-5 rounded-2xl bg-[#0d0e14]/40 backdrop-blur-xl border border-white/10 shadow-xl">
      {/* 1. TOP-CENTER NAVIGATION TABS */}
      <div className="w-full flex items-center justify-center pt-0.5 pb-1">
        <div className="flex items-center gap-1.5 bg-white/5 p-1 rounded-2xl border border-white/10 shadow-xs">
          {(
            [
              { id: "forecast", label: "Balance Trend" },
              { id: "calendar", label: "Payments Calendar" },
              { id: "expenses", label: "Expense Breakdown" },
              { id: "cashflow", label: "Cash Flow" },
            ] as const
          ).map((t) => (
            <Button
              key={t.id}
              variant="tertiary"
              onPress={() => setActiveView(t.id)}
              className={`px-3.5 py-1.5 h-auto rounded-xl text-xs font-bold transition-all cursor-pointer shadow-none ${
                activeView === t.id
                  ? "bg-primary text-primary-foreground shadow-xs"
                  : "bg-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              {t.label}
            </Button>
          ))}

          {/* Dropdown for More Views */}
          <Dropdown>
            <Button
              aria-label="More views"
              variant="tertiary"
              className={`px-3 py-1.5 h-auto rounded-xl text-xs font-bold flex items-center gap-1.5 transition-all cursor-pointer shadow-none ${
                [
                  "pace",
                  "subscriptions",
                  "goals",
                  "wants_needs",
                  "merchants",
                  "allowance",
                ].includes(activeView)
                  ? "bg-primary/20 text-primary border border-primary/30"
                  : "bg-transparent text-muted-foreground hover:text-foreground"
              }`}
            >
              <span>More</span>
              <ChevronDown size={13} />
            </Button>
            <Dropdown.Popover className="bg-[#0d0e14]/95 backdrop-blur-2xl border border-white/10 shadow-2xl rounded-xl p-1.5 z-50 w-52">
              <Dropdown.Menu
                onAction={(key) => setActiveView(key as CardViewMode)}
                className="text-xs"
              >
                <Dropdown.Item id="pace">Spending Pace</Dropdown.Item>
                <Dropdown.Item id="subscriptions">Active Subscriptions</Dropdown.Item>
                <Dropdown.Item id="goals">Savings Goals</Dropdown.Item>
                <Dropdown.Item id="wants_needs">Needs vs. Wants (50/30/20)</Dropdown.Item>
                <Dropdown.Item id="merchants">Top Merchants</Dropdown.Item>
                <Dropdown.Item id="allowance">Daily Spending Allowance</Dropdown.Item>
              </Dropdown.Menu>
            </Dropdown.Popover>
          </Dropdown>
        </div>
      </div>

      {/* 2. HEADINGS & VIEW-SPECIFIC CONTROLS */}
      <div className="flex flex-col sm:flex-row sm:items-center justify-between gap-3 border-b border-white/10 dark:border-white/5 pb-3">
        <div>
          <div className="flex items-center gap-2">
            <h3 className="text-base font-bold text-foreground font-sans">
              {viewMeta.title}
            </h3>
            <span className="text-[10px] font-bold px-2 py-0.5 rounded-full bg-primary/10 text-primary border border-primary/20 uppercase tracking-wider">
              {viewMeta.badge}
            </span>
          </div>
          <p className="text-xs text-muted-foreground mt-0.5 font-sans">
            {viewMeta.subtitle}
          </p>
        </div>

        {/* View-Specific Actions (Forecast Lookback & Forecast Toggle) */}
        {activeView === "forecast" && (
          <div className="flex flex-wrap items-center gap-2">
            {/* Past Lookback Pills */}
            <div className="flex items-center gap-1 bg-white/5 p-1 rounded-xl border border-white/10">
              <span className="text-[10px] text-muted-foreground font-semibold px-1.5">
                Past:
              </span>
              {([30, 60, 90] as const).map((days) => (
                <Button
                  key={days}
                  size="sm"
                  variant="tertiary"
                  onPress={() => setPastLookbackDays(days)}
                  className={`px-2 py-0.5 h-6 min-w-0 rounded-lg text-[10px] font-bold transition-all cursor-pointer shadow-none ${
                    pastLookbackDays === days
                      ? "bg-blue-600 text-white"
                      : "bg-transparent text-muted-foreground hover:text-foreground"
                  }`}
                >
                  {days}d
                </Button>
              ))}
            </div>

            {/* Forward Forecast Horizon Pills */}
            {isForecastExtended && (
              <div className="flex items-center gap-1 bg-white/5 p-1 rounded-xl border border-white/10">
                <span className="text-[10px] text-muted-foreground font-semibold px-1.5">
                  Ahead:
                </span>
                {([30, 60, 90] as const).map((days) => (
                  <Button
                    key={days}
                    size="sm"
                    variant="tertiary"
                    onPress={() => handleHorizonChange(days)}
                    className={`px-2 py-0.5 h-6 min-w-0 rounded-lg text-[10px] font-bold transition-all cursor-pointer shadow-none ${
                      horizonDays === days
                        ? "bg-purple-600 text-white"
                        : "bg-transparent text-muted-foreground hover:text-foreground"
                    }`}
                  >
                    {days}d
                  </Button>
                ))}
              </div>
            )}

            {/* Extend Forecast Button */}
            <Button
              size="sm"
              onPress={handleToggleForecast}
              isPending={isFetchingBackendForecast}
              className={`text-xs font-bold rounded-xl px-3 h-8 transition-all flex items-center gap-1.5 cursor-pointer shadow-sm ${
                isForecastExtended
                  ? "bg-purple-600 text-white hover:bg-purple-700"
                  : "bg-white/10 hover:bg-white/15 text-foreground border border-white/10"
              }`}
            >
              <Sparkles size={13} />
              <span>
                {isForecastExtended ? "Hide Forecast" : "Extend Forecast"}
              </span>
            </Button>
          </div>
        )}
      </div>

      {/* --------------------------------------------------------------------- */}
      {/* 3-Box Stat Row (Adapts cleanly to active view) */}
      {/* --------------------------------------------------------------------- */}
      <div className="grid grid-cols-2 sm:grid-cols-3 gap-3 py-1">
        {activeView === "forecast" && (
          <>
            <div className="p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Current Balance
              </span>
              <span className="text-base font-bold text-foreground font-mono">
                {formatCurrency(currentBalance, activeCurrency)}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Safety Floor Cushion
              </span>
              <span className="text-base font-bold text-foreground font-mono">
                {formatCurrency(safetyCushion, activeCurrency)}
              </span>
            </div>
            <div className="col-span-2 sm:col-span-1 p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                {isForecastExtended
                  ? `Projected (${horizonDays}d)`
                  : "Daily Average Spend"}
              </span>
              <span
                className={`text-base font-bold font-mono ${
                  isForecastExtended && isBelowCushion
                    ? "text-red-500"
                    : "text-foreground"
                }`}
              >
                {isForecastExtended
                  ? formatCurrency(projectedBalance, activeCurrency)
                  : `${formatCurrency(avgDailySpend, activeCurrency)} / day`}
              </span>
            </div>
          </>
        )}

        {activeView === "calendar" && (
          <>
            <div className="p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Selected Date Due
              </span>
              <span className="text-base font-bold text-foreground font-mono">
                {formatCurrency(
                  selectedDateEvents.reduce((acc, e) => acc + e.amount, 0),
                  activeCurrency,
                )}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Total Bills This Month
              </span>
              <span className="text-base font-bold text-amber-500 font-mono">
                {formatCurrency(totalBillsMonth, activeCurrency)}
              </span>
            </div>
            <div className="col-span-2 sm:col-span-1 p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Subscriptions Total
              </span>
              <span className="text-base font-bold text-purple-400 font-mono">
                {formatCurrency(totalSubsMonth, activeCurrency)}
              </span>
            </div>
          </>
        )}

        {activeView === "expenses" && (
          <>
            <div className="p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Total Outflow
              </span>
              <span className="text-base font-bold text-foreground font-mono">
                {formatCurrency(totalExpenseAmount, activeCurrency)}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Top Category
              </span>
              <span className="text-base font-bold text-foreground truncate block font-sans">
                {categoryData[0]?.name || "None"}
              </span>
            </div>
            <div className="col-span-2 sm:col-span-1 p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Categories Tracked
              </span>
              <span className="text-base font-bold text-foreground font-mono">
                {categoryData.length} Active
              </span>
            </div>
          </>
        )}

        {activeView === "cashflow" && (
          <>
            <div className="p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Money In (This Month)
              </span>
              <span className="text-base font-bold text-emerald-500 font-mono">
                +{formatCurrency(cashflowMonths.thisMonthIn, activeCurrency)}
              </span>
            </div>
            <div className="p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Money Out (This Month)
              </span>
              <span className="text-base font-bold text-red-400 font-mono">
                -{formatCurrency(cashflowMonths.thisMonthOut, activeCurrency)}
              </span>
            </div>
            <div className="col-span-2 sm:col-span-1 p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Net Saved
              </span>
              <span className="text-base font-bold text-emerald-400 font-mono">
                +{formatCurrency(cashflowMonths.thisMonthNet, activeCurrency)}
              </span>
            </div>
          </>
        )}

        {["pace", "subscriptions", "goals", "wants_needs", "merchants", "allowance"].includes(
          activeView,
        ) && (
          <>
            <div className="p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Spending Growth
              </span>
              <span className="text-base font-bold text-foreground font-mono">
                {cashflowMonths.spendDeltaPct >= 0 ? "+" : ""}
                {cashflowMonths.spendDeltaPct}% vs Last Mo
              </span>
            </div>
            <div className="p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Active Subscriptions
              </span>
              <span className="text-base font-bold text-purple-400 font-mono">
                {formatCurrency(totalSubsMonth, activeCurrency)}/mo
              </span>
            </div>
            <div className="col-span-2 sm:col-span-1 p-3 rounded-xl bg-white/40 dark:bg-neutral-800/40 border border-white/10 dark:border-white/5">
              <span className="text-[10px] uppercase tracking-wider text-muted-foreground font-semibold block mb-0.5">
                Daily Safe Spend
              </span>
              <span className="text-base font-bold text-emerald-400 font-mono">
                {formatCurrency(allowanceInfo.safeDailySpend, activeCurrency)} / day
              </span>
            </div>
          </>
        )}
      </div>

      {/* --------------------------------------------------------------------- */}
      {/* Main Visual Display (Expanded min-h-[350px] container) */}
      {/* --------------------------------------------------------------------- */}
      <div className="w-full min-h-[350px] h-auto md:h-[350px] relative mt-1">
        {/* VIEW 1: BALANCE TREND & FORECAST */}
        {activeView === "forecast" && (
          <>
            {isTrendLoading ? (
              <div className="w-full h-full flex flex-col justify-end gap-2 p-2">
                <Skeleton className="h-4/5 w-full rounded-xl" />
              </div>
            ) : (
              <CoreChartEngine config={forecastChartConfig} />
            )}
          </>
        )}

        {/* VIEW 2: PAYMENTS & BILLS CALENDAR (50/50 SPLIT: EQUAL WIDTH CALENDAR & CONTINUOUS DATA CARD) */}
        {activeView === "calendar" && (
          <div className="grid grid-cols-1 md:grid-cols-2 gap-5 w-full h-full">
            {/* Left Column: Calendar Card (Equal 50% width) */}
            <div className="w-full h-full flex flex-col p-4 bg-white/5 rounded-2xl border border-white/10 overflow-hidden">
              <Calendar
                aria-label="Payment dates calendar"
                minValue={minCalendarDate}
                maxValue={maxCalendarDate}
                focusedValue={focusedValue}
                onFocusChange={setFocusedValue}
                value={selectedCalendarDate}
                onChange={(d) => setSelectedCalendarDate(d as CalendarDate)}
                className="w-full h-full flex flex-col justify-between bg-transparent border-0 shadow-none text-foreground relative"
              >
                <Calendar.Header className="flex items-center justify-between pb-2 mb-1.5 border-b border-white/5 shrink-0">
                  {/* Compact Year Picker Trigger styled to match card aesthetic */}
                  <Calendar.YearPickerTrigger className="flex-none grow-0 w-fit max-w-fit px-2.5 h-7 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-foreground text-xs font-semibold flex items-center gap-1.5 cursor-pointer transition-colors outline-none shrink-0">
                    <Calendar.YearPickerTriggerHeading className="text-xs font-bold text-foreground" />
                    <Calendar.YearPickerTriggerIndicator className="text-muted-foreground w-3.5 h-3.5" />
                  </Calendar.YearPickerTrigger>

                  {/* Today Button & Navigation Controls on the same row */}
                  <div className="flex items-center gap-1.5 shrink-0">
                    <button
                      type="button"
                      onClick={handleToday}
                      className="h-7 px-2.5 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-xs font-semibold text-muted-foreground hover:text-foreground cursor-pointer transition-colors"
                    >
                      Today
                    </button>
                    <Calendar.NavButton
                      slot="previous"
                      className="h-7 w-7 p-0 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-muted-foreground hover:text-foreground flex items-center justify-center cursor-pointer transition-colors"
                    />
                    <Calendar.NavButton
                      slot="next"
                      className="h-7 w-7 p-0 rounded-lg bg-white/5 hover:bg-white/10 border border-white/10 text-muted-foreground hover:text-foreground flex items-center justify-center cursor-pointer transition-colors"
                    />
                  </div>
                </Calendar.Header>

                <Calendar.Grid className="w-full border-collapse">
                  <Calendar.GridHeader>
                    {(day) => (
                      <Calendar.HeaderCell className="text-xs font-bold text-muted-foreground/70 pb-2 text-center uppercase tracking-wider font-sans">
                        {day}
                      </Calendar.HeaderCell>
                    )}
                  </Calendar.GridHeader>
                  <Calendar.GridBody className="text-sm">
                    {(date) => (
                      <Calendar.Cell
                        date={date}
                        className="p-0.5 text-center cursor-pointer bg-transparent data-[selected=true]:bg-transparent data-[hovered=true]:bg-transparent hover:bg-transparent shadow-none data-[selected=true]:shadow-none data-[focus-visible=true]:ring-0 outline-none"
                      >
                        {({ formattedDate, isSelected, isOutsideMonth }) => {
                          const isTodayDate = date.compare(todayDate) === 0;

                          const hasBill = calendarEvents.some(
                            (e) => e.dayOfMonth === date.day && e.type === "bill",
                          );
                          const hasSub = calendarEvents.some(
                            (e) => e.dayOfMonth === date.day && e.type === "subscription",
                          );
                          const hasIncome = calendarEvents.some(
                            (e) => e.dayOfMonth === date.day && e.type === "income",
                          );

                          return (
                            <div
                              onClick={() => handleDateCellClick(date)}
                              className={`relative flex flex-col items-center justify-center w-9 h-9 sm:w-10 sm:h-10 mx-auto rounded-full transition-all bg-transparent ${
                                isSelected
                                  ? "border-2 border-primary ring-2 ring-primary/20 text-foreground font-bold shadow-xs bg-transparent"
                                  : isTodayDate
                                  ? "border border-white/30 text-foreground font-semibold bg-transparent"
                                  : !isOutsideMonth
                                  ? "border-2 border-transparent text-foreground hover:border-white/20 bg-transparent"
                                  : "border-2 border-transparent text-muted-foreground/35 hover:text-muted-foreground hover:border-white/10 bg-transparent"
                              }`}
                            >
                              <span className="text-sm font-mono font-semibold leading-none">
                                {formattedDate}
                              </span>
                              {(hasBill || hasSub || hasIncome) && (
                                <div className="flex items-center gap-0.5 absolute bottom-1">
                                  {hasBill && (
                                    <div className="w-1.5 h-1.5 rounded-full bg-amber-500" />
                                  )}
                                  {hasSub && (
                                    <div className="w-1.5 h-1.5 rounded-full bg-purple-500" />
                                  )}
                                  {hasIncome && (
                                    <div className="w-1.5 h-1.5 rounded-full bg-emerald-500" />
                                  )}
                                </div>
                              )}
                            </div>
                          );
                        }}
                      </Calendar.Cell>
                    )}
                  </Calendar.GridBody>
                </Calendar.Grid>

                {/* Year Picker Grid Overlay: Gracefully contained to exact calendar dimensions */}
                <Calendar.YearPickerGrid className="p-1.5 bg-[#0d0e14]/95 backdrop-blur-2xl border border-white/10 rounded-xl shadow-xl">
                  <Calendar.YearPickerGridBody>
                    {({ year }) => (
                      <Calendar.YearPickerCell
                        year={year}
                        className="h-8 px-2 w-full flex items-center justify-center mx-auto text-xs font-mono text-foreground rounded-lg hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                      />
                    )}
                  </Calendar.YearPickerGridBody>
                </Calendar.YearPickerGrid>
              </Calendar>
            </div>

            {/* Right Column: Selected Date Data Card (Continuously Displayed, Equal 50% width) */}
            <div className="w-full h-full flex flex-col p-4 bg-white/5 rounded-2xl border border-white/10 overflow-hidden">
              <div className="flex items-center justify-between pb-3 border-b border-white/10 shrink-0">
                <div>
                  <h4 className="text-sm font-bold text-foreground font-sans">
                    {selectedCalendarDate
                      .toDate(getLocalTimeZone())
                      .toLocaleDateString("en-GB", {
                        weekday: "long",
                        day: "numeric",
                        month: "long",
                        year: "numeric",
                      })}
                  </h4>
                  <span className="text-xs text-muted-foreground">
                    {selectedDateEvents.length > 0
                      ? `${selectedDateEvents.length} scheduled item${selectedDateEvents.length === 1 ? "" : "s"}`
                      : "No financial activity scheduled"}
                  </span>
                </div>
                {selectedDateEvents.length > 0 && (
                  <span className="text-xs font-bold text-foreground font-mono bg-white/5 px-2.5 py-1 rounded-lg border border-white/10">
                    Total: {formatCurrency(
                      selectedDateEvents.reduce((acc, e) => (e.type === "income" ? acc : acc + e.amount), 0),
                      activeCurrency,
                    )}
                  </span>
                )}
              </div>

              {/* Content or "No financial activity" */}
              <div className="flex-1 overflow-y-auto scrollbar-hide py-2.5 flex flex-col gap-2">
                {selectedDateEvents.length === 0 ? (
                  <div className="flex-1 flex flex-col items-center justify-center text-center p-6">
                    <CalendarIcon size={24} className="text-muted-foreground/30 mb-2" />
                    <p className="text-sm font-semibold text-foreground font-sans">
                      No financial activity
                    </p>
                    <span className="text-xs text-muted-foreground mt-1">
                      There are no bills, subscriptions, or income entries for this date.
                    </span>
                  </div>
                ) : (
                  selectedDateEvents.map((evt) => (
                    <div
                      key={evt.id}
                      className="p-3 rounded-xl bg-[#0d0e14]/70 border border-white/10 flex items-center justify-between gap-3"
                    >
                      <div className="flex items-center gap-3 min-w-0">
                        <div
                          className={`w-8 h-8 rounded-lg flex items-center justify-center shrink-0 ${
                            evt.type === "bill"
                              ? "bg-amber-500/15 text-amber-500"
                              : evt.type === "subscription"
                              ? "bg-purple-500/15 text-purple-400"
                              : "bg-emerald-500/15 text-emerald-400"
                          }`}
                        >
                          {evt.type === "bill" ? (
                            <CreditCard size={15} />
                          ) : evt.type === "subscription" ? (
                            <Layers size={15} />
                          ) : (
                            <ArrowDownRight size={15} />
                          )}
                        </div>
                        <div className="min-w-0">
                          <span className="text-xs font-bold text-foreground block truncate font-sans">
                            {evt.title}
                          </span>
                          <div className="flex items-center gap-2 mt-0.5">
                            <span className="text-[10px] text-muted-foreground uppercase font-semibold">
                              {evt.type}
                            </span>
                            <span className="text-[10px] text-muted-foreground font-mono">
                              {evt.status}
                            </span>
                          </div>
                        </div>
                      </div>
                      <span
                        className={`text-sm font-bold font-mono shrink-0 ${
                          evt.type === "income" ? "text-emerald-400" : "text-foreground"
                        }`}
                      >
                        {evt.type === "income" ? "+" : "-"}
                        {formatCurrency(evt.amount, activeCurrency)}
                      </span>
                    </div>
                  ))
                )}
              </div>
            </div>
          </div>
        )}

        {/* VIEW 3: EXPENSE BREAKDOWN (DONUT / PIE CHART) */}
        {activeView === "expenses" && (
          categoryData.length === 0 ? (
            <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center">
              <PieChart size={24} className="text-muted-foreground/30 mb-2" />
              <p className="text-sm font-semibold text-foreground font-sans">
                No expense data recorded
              </p>
              <span className="text-xs text-muted-foreground mt-1">
                Category spending distributions will appear as verified transactions are processed.
              </span>
            </div>
          ) : (
            <div className="w-full h-full flex flex-col sm:flex-row items-center gap-4">
              {/* Left: Donut Chart with center cutout */}
              <div className="w-full sm:w-[48%] h-full relative flex items-center justify-center">
                <CoreChartEngine config={expensesPieConfig} />
                <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none">
                  <span className="text-[10px] uppercase font-bold text-muted-foreground font-sans">
                    Total
                  </span>
                  <span className="text-sm font-bold text-foreground font-mono">
                    {formatCurrency(totalExpenseAmount, activeCurrency)}
                  </span>
                </div>
              </div>

              {/* Right: Ranked Category List */}
              <div className="w-full sm:w-[52%] h-full flex flex-col gap-1.5 overflow-y-auto scrollbar-hide pr-1">
                {categoryData.slice(0, 6).map((cat, idx) => {
                  const pct =
                    totalExpenseAmount > 0
                      ? Math.round((cat.value / totalExpenseAmount) * 100)
                      : 0;
                  const dotColor = pieColors[idx % pieColors.length];
                  return (
                    <div
                      key={cat.name}
                      className="flex items-center justify-between p-2 rounded-lg bg-white/5 border border-white/5 text-xs"
                    >
                      <div className="flex items-center gap-2 min-w-0">
                        <div
                          className="w-2.5 h-2.5 rounded-full shrink-0"
                          style={{ backgroundColor: dotColor }}
                        />
                        <span className="font-semibold text-foreground truncate font-sans">
                          {cat.name}
                        </span>
                      </div>
                      <div className="flex items-center gap-2 shrink-0 font-mono">
                        <span className="text-foreground font-bold">
                          {formatCurrency(cat.value, activeCurrency)}
                        </span>
                        <span className="text-[10px] text-muted-foreground">
                          {pct}%
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )
        )}

        {/* VIEW 4: CASH FLOW (MONTHLY IN VS OUT BARS) */}
        {activeView === "cashflow" && (
          <CoreChartEngine config={cashflowBarConfig} />
        )}

        {/* VIEW 5: SPENDING PACE COMPARISON */}
        {activeView === "pace" && (
          <CoreChartEngine config={paceChartConfig} />
        )}

        {/* VIEW 6: ACTIVE SUBSCRIPTIONS */}
        {activeView === "subscriptions" && (
          subscriptionsList.length === 0 ? (
            <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center">
              <Layers size={24} className="text-muted-foreground/30 mb-2" />
              <p className="text-sm font-semibold text-foreground font-sans">
                No active subscriptions detected
              </p>
              <span className="text-xs text-muted-foreground mt-1">
                Recurring commitments and memberships will appear here once identified.
              </span>
            </div>
          ) : (
            <div className="grid grid-cols-2 sm:grid-cols-3 gap-2.5 h-full overflow-y-auto scrollbar-hide">
              {subscriptionsList.map((sub, idx) => (
                <div
                  key={`${sub.name}-${idx}`}
                  className="p-3 rounded-xl bg-white/5 border border-white/10 flex flex-col justify-between"
                >
                  <div>
                    <span className="text-xs font-bold text-foreground block truncate font-sans">
                      {sub.name}
                    </span>
                    <span className="text-[10px] text-muted-foreground font-sans">
                      {sub.renewal}
                    </span>
                  </div>
                  <span className="text-sm font-bold text-purple-400 font-mono mt-2">
                    {formatCurrency(sub.cost, activeCurrency)} / mo
                  </span>
                </div>
              ))}
            </div>
          )
        )}

        {/* VIEW 7: SAVINGS GOALS */}
        {activeView === "goals" && (
          savingsGoals.length === 0 ? (
            <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center">
              <Target size={24} className="text-muted-foreground/30 mb-2" />
              <p className="text-sm font-semibold text-foreground font-sans">
                No savings goals configured
              </p>
              <span className="text-xs text-muted-foreground mt-1">
                Virtual capital buckets and target reserves will display progress here.
              </span>
            </div>
          ) : (
            <div className="flex flex-col gap-2.5 h-full overflow-y-auto scrollbar-hide">
              {savingsGoals.map((g, idx) => (
                <div
                  key={`${g.name}-${idx}`}
                  className="p-3 rounded-xl bg-white/5 border border-white/10 flex flex-col gap-1.5"
                >
                  <div className="flex items-center justify-between text-xs font-sans">
                    <span className="font-bold text-foreground">{g.name}</span>
                    <span className="text-[11px] font-mono text-muted-foreground">
                      {formatCurrency(g.current, activeCurrency)} of{" "}
                      {formatCurrency(g.target, activeCurrency)} ({g.pct}%)
                    </span>
                  </div>
                  <div className="w-full h-1.5 rounded-full bg-white/10 overflow-hidden">
                    <div
                      className="h-full bg-emerald-500 rounded-full transition-all"
                      style={{ width: `${g.pct}%` }}
                    />
                  </div>
                </div>
              ))}
            </div>
          )
        )}

        {/* VIEW 8: 50/30/20 NEEDS VS WANTS */}
        {activeView === "wants_needs" && (
          <div className="flex flex-col justify-around h-full p-2">
            {wantsNeedsData.map((item) => (
              <div key={item.label} className="flex flex-col gap-1">
                <div className="flex items-center justify-between text-xs font-sans">
                  <span className="font-semibold text-foreground">{item.label}</span>
                  <span className="font-bold text-foreground font-mono">
                    {formatCurrency(item.amount, activeCurrency)} ({item.pct}%)
                  </span>
                </div>
                <div className="w-full h-2 rounded-full bg-white/10 overflow-hidden">
                  <div
                    className={`h-full ${item.color} rounded-full`}
                    style={{ width: `${item.pct}%` }}
                  />
                </div>
              </div>
            ))}
          </div>
        )}

        {/* VIEW 9: TOP MERCHANTS */}
        {activeView === "merchants" && (
          topMerchants.length === 0 ? (
            <div className="w-full h-full flex flex-col items-center justify-center p-6 text-center">
              <ShoppingBag size={24} className="text-muted-foreground/30 mb-2" />
              <p className="text-sm font-semibold text-foreground font-sans">
                No merchant transactions recorded
              </p>
              <span className="text-xs text-muted-foreground mt-1">
                Highest outflow stores and companies will rank here as transactions settle.
              </span>
            </div>
          ) : (
            <div className="grid grid-cols-1 sm:grid-cols-2 gap-2 h-full overflow-y-auto scrollbar-hide">
              {topMerchants.map((m) => (
                <div
                  key={m.name}
                  className="p-2.5 rounded-xl bg-white/5 border border-white/10 flex items-center justify-between"
                >
                  <div className="flex items-center gap-2.5 min-w-0">
                    <span className="w-5 h-5 rounded-md bg-white/10 flex items-center justify-center text-[10px] font-bold text-muted-foreground font-mono">
                      #{m.rank}
                    </span>
                    <div className="min-w-0">
                      <span className="text-xs font-bold text-foreground block truncate font-sans">
                        {m.name}
                      </span>
                      <span className="text-[10px] text-muted-foreground font-sans">
                        {m.count}
                      </span>
                    </div>
                  </div>
                  <span className="text-xs font-bold text-foreground font-mono shrink-0">
                    {formatCurrency(m.amount, activeCurrency)}
                  </span>
                </div>
              ))}
            </div>
          )
        )}

        {/* VIEW 10: DAILY ALLOWANCE & PAYDAY COUNTDOWN */}
        {activeView === "allowance" && (
          <div className="flex flex-col sm:flex-row items-center justify-around h-full p-4 gap-4">
            <div className="flex flex-col items-center justify-center text-center">
              <span className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider font-sans">
                Safe Daily Spend
              </span>
              <span className="text-3xl font-bold text-emerald-400 font-mono mt-1">
                {formatCurrency(allowanceInfo.safeDailySpend, activeCurrency)}
              </span>
              <span className="text-xs text-muted-foreground mt-0.5 font-sans">
                Per day for the next {allowanceInfo.daysRemaining} {allowanceInfo.daysRemaining === 1 ? "day" : "days"}
              </span>
            </div>
            <div className="w-px h-24 bg-white/10 hidden sm:block" />
            <div className="flex flex-col items-center justify-center text-center">
              <span className="text-[10px] uppercase font-bold text-muted-foreground tracking-wider font-sans">
                Next Paycheck Due
              </span>
              <span className="text-3xl font-bold text-foreground font-mono mt-1">
                {allowanceInfo.daysRemaining} {allowanceInfo.daysRemaining === 1 ? "Day" : "Days"}
              </span>
              <span className="text-xs text-muted-foreground mt-0.5 font-sans">
                Expected {allowanceInfo.expectedDateStr}
              </span>
            </div>
          </div>
        )}
      </div>

      {/* --------------------------------------------------------------------- */}
      {/* Bottom Legend & Status Row */}
      {/* --------------------------------------------------------------------- */}
      <div className="flex items-center justify-between text-xs text-muted-foreground pt-2.5 border-t border-white/10 dark:border-white/5 font-sans">
        {activeView === "forecast" && (
          <>
            <div className="flex items-center gap-3.5">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-0.5 bg-blue-500 rounded-full" />
                <span>Past Balance</span>
              </div>
              {isForecastExtended && (
                <div className="flex items-center gap-1.5">
                  <div className="w-2.5 h-0.5 bg-purple-500 border-dashed rounded-full" />
                  <span>Projected</span>
                </div>
              )}
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-0.5 bg-red-500/70 rounded-full" />
                <span>Floor ({formatCurrency(safetyCushion, activeCurrency)})</span>
              </div>
            </div>

            {isForecastExtended && (
              <div className="flex items-center gap-1 text-xs font-semibold text-foreground/80">
                <ShieldCheck size={13} className="text-emerald-500" />
                <span>
                  {isBelowCushion
                    ? "Projected Below Safety Floor"
                    : "Projected Above Safety Floor"}
                </span>
              </div>
            )}
          </>
        )}

        {activeView === "calendar" && (
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-amber-500" />
                <span>Utility Bills</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-purple-500" />
                <span>Subscriptions</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2 h-2 rounded-full bg-emerald-500" />
                <span>Payday</span>
              </div>
            </div>
            <span className="text-[11px] text-muted-foreground">
              Click any date to inspect details
            </span>
          </div>
        )}

        {activeView === "expenses" && (
          <div className="flex items-center justify-between w-full">
            <span>
              {categoryData.length > 0 && totalExpenseAmount > 0
                ? `Top category accounts for ${Math.round((categoryData[0].value / totalExpenseAmount) * 100)}% of recorded outflows`
                : "No recorded expense distribution"}
            </span>
            <span className="font-mono text-foreground font-semibold">
              {categoryData.length} {categoryData.length === 1 ? "Category" : "Categories"}
            </span>
          </div>
        )}

        {activeView === "cashflow" && (
          <div className="flex items-center justify-between w-full">
            <div className="flex items-center gap-3">
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2 rounded-xs bg-emerald-500" />
                <span>Money In</span>
              </div>
              <div className="flex items-center gap-1.5">
                <div className="w-2.5 h-2 rounded-xs bg-red-400" />
                <span>Money Out</span>
              </div>
            </div>
            <span>
              Savings retention:{" "}
              {cashflowMonths.thisMonthIn > 0
                ? Math.max(0, Math.round((cashflowMonths.thisMonthNet / cashflowMonths.thisMonthIn) * 100))
                : 0}
              % of income
            </span>
          </div>
        )}

        {["pace", "subscriptions", "goals", "wants_needs", "merchants", "allowance"].includes(
          activeView,
        ) && (
          <div className="flex items-center justify-between w-full">
            <span>All estimates grounded in verified account transactions</span>
            <span className="text-[11px] font-semibold text-emerald-400">
              System Operational
            </span>
          </div>
        )}
      </div>
    </div>
  );
};

export const SpendingForecastCard = FinancialAnalysisCard;

