"use client";

import React, { useState, useMemo, useEffect, useCallback } from "react";
import { ChevronDown, Activity, Calendar as CalendarIcon, Target, Settings } from "lucide-react";
import CoreChartEngine from "../../internal/ChartEngine";
import type { Selection } from "@heroui/react";
import {
  Card,
  Dropdown,
  Label,
  Description,
  Skeleton,
  DatePicker,
  DateField,
  Calendar,
  Badge,
  ToggleButton,
  ToggleButtonGroup,
  CloseButton,
  ProgressBar,
  Spinner,
  Button
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { today, getLocalTimeZone, DateValue } from "@internationalized/date";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { useSpendingTrends, usePersistedState, useTransactions, usePersistedDate } from "@/lib/hooks";
import { apiFetch } from "@/lib/api";
import WidgetFlipCard, { FlipButton } from "../../internal/FlipCard";
import SimulationControlsModal, { SimulationOverrides } from "@/app/(protected)/_components/modals/SimulationControlsModal";
import { useRouter } from "next/navigation";
import { Account, BankChartData } from "@/types";
import { cn } from "@/lib/utils";
import { WidgetContext } from "../../../home/DashboardClient";

interface SpendingTrendWidgetProps {
  initialData?: BankChartData[];
}

export default function SpendingTrendWidgetClient({
  initialData,
}: SpendingTrendWidgetProps) {
  const router = useRouter();
  const { onRemove, instanceId } = React.useContext(WidgetContext);
  const { accounts, createNewSession } = useBudAI();

  const [selectedAccountId, setSelectedAccountId] = usePersistedState<string>(
    `trends_account${instanceId ? `-${instanceId}` : ""}`,
    accounts[0]?.account_id || "",
  );

  React.useEffect(() => {
    if (accounts.length > 0) {
      if (!selectedAccountId || selectedAccountId.startsWith("react-aria-") || !accounts.find(a => a.account_id === selectedAccountId)) {
        setSelectedAccountId(accounts[0].account_id);
      }
    }
  }, [accounts, selectedAccountId, setSelectedAccountId]);
  const [granularity, setGranularity] = usePersistedState<string>(`trends_granularity_${instanceId || ""}`, "monthly");

  const [startDate, setStartDate] = usePersistedDate(`trends_start_${instanceId || ""}`, 
    today(getLocalTimeZone()).subtract({ months: 6 }),
  );

  const [endDate, setEndDate] = usePersistedDate(`trends_end_${instanceId || ""}`, 
    today(getLocalTimeZone()),
  );

  const fromStr = startDate
    ? `${startDate.year}-${String(startDate.month).padStart(2, "0")}-${String(startDate.day).padStart(2, "0")}`
    : "";
  const toStr = endDate
    ? `${endDate.year}-${String(endDate.month).padStart(2, "0")}-${String(endDate.day).padStart(2, "0")}`
    : "";

  const {
    data: transactions = [],
    isFetching: isTransactionsFetching,
  } = useTransactions(selectedAccountId, fromStr, toStr);

  const hasTransactions = transactions.length > 0;

  const {
    data: chartPayload = [],
    isLoading: isChartLoading,
    isFetching: isChartFetching,
  } = useSpendingTrends(
    selectedAccountId,
    fromStr,
    toStr,
    granularity as "daily" | "weekly" | "monthly",
    granularity === "monthly" ? initialData : undefined,
    hasTransactions,
  );

  const [isModalOpen, setIsModalOpen] = useState(false);
  const [simulationOverrides, setSimulationOverrides] = useState<SimulationOverrides>({
    discipline_multiplier: 1.0,
    drift_adjustment: 0.0,
    macro_environment: "Stable",
    stress_test_active: false,
    days: 30,
  });

  const [expenseForecast, setExpenseForecast] = useState<any>(null);
  const [isForecastLoading, setIsForecastLoading] = useState(false);

  const fetchExpenseForecast = useCallback(async () => {
    if (!selectedAccountId) return;
    setIsForecastLoading(true);
    try {
      const expenseRes = await apiFetch(
        "/api/media/execute",
        {
          method: "POST",
          body: JSON.stringify({
            tool_name: "generate_expense_forecast",
            parameters: {
              account_id: selectedAccountId,
              ...simulationOverrides,
            },
          }),
        },
        true
      );
      if (expenseRes.ok) {
        const result = await expenseRes.json() as any;
        setExpenseForecast(result.data);
      }
    } catch (error) {
      console.error("Forecast Fetch Error:", error);
    } finally {
      setIsForecastLoading(false);
    }
  }, [selectedAccountId, simulationOverrides]);

  useEffect(() => {
    fetchExpenseForecast();
  }, [fetchExpenseForecast]);

  const combinedPayload = useMemo(() => {
    if (!chartPayload || chartPayload.length === 0) return chartPayload;
    if (!expenseForecast) return chartPayload;

    const series = (expenseForecast && typeof expenseForecast === 'object' && 'series' in expenseForecast) ? expenseForecast.series : (Array.isArray(expenseForecast) ? expenseForecast : [expenseForecast]);
    const timeline = (expenseForecast && typeof expenseForecast === 'object' && 'timeline' in expenseForecast) ? expenseForecast.timeline : [];

    const baseData = chartPayload[0]?.data || [];
    const forecastData = series[0]?.data || [];

    const mappedForecast = forecastData.map((pt: any) => {
      const date = pt.Day || pt.Date || pt.Month || "";
      const val = Number(pt["Projected Daily Spend (GBP)"] || pt["Projected Spend"] || pt["spend"] || 0);
      return {
        Date: date,
        Month: date,
        Category: "Forecast",
        Amount: val,
        currency: pt.currency || "GBP",
      };
    });

    return {
      series: [
        {
          ...chartPayload[0],
          data: [...baseData, ...mappedForecast]
        }
      ],
      timeline
    };
  }, [chartPayload, expenseForecast]);

  const isInitialLoading = isChartLoading || (isTransactionsFetching && !hasTransactions) || (isForecastLoading && !expenseForecast);
  const isFetching = isChartFetching || isTransactionsFetching || isForecastLoading;

  const totalExpenses = useMemo(() => {
    let total = 0;
    if (Array.isArray(chartPayload)) {
      chartPayload.forEach((bank: BankChartData) => {
        if (Array.isArray(bank.data)) {
          bank.data.forEach((point: Record<string, string | number>) => {
            total += Number(point.Amount || point.amount || 0);
          });
        }
      });
    }
    return total;
  }, [chartPayload]);

  const config = useMemo(() => {
    const isArray = Array.isArray(combinedPayload);
    const series = isArray ? combinedPayload : (combinedPayload as any).series;
    
    if (!startDate || !endDate || !series || series.length === 0) return null;

    const chartType =
      granularity === "daily"
        ? "historical_daily"
        : granularity === "weekly"
          ? "historical_weekly"
          : "historical_monthly";

    return buildChartConfig(
      chartType,
      combinedPayload as any,
      {
        bank_name_or_id: selectedAccountId,
        from_date: fromStr,
        to_date: toStr,
      },
      `${granularity.charAt(0).toUpperCase() + granularity.slice(1)} Spending Trends`,
      { disableAnimation: true },
    );
  }, [
    combinedPayload,
    startDate,
    endDate,
    selectedAccountId,
    fromStr,
    toStr,
    granularity,
  ]);

  const handleDiscuss = () => {
    const sessionId = createNewSession("Spending Trend Deep Dive", {
      type: "spending_trend",
      accountId: selectedAccountId,
      data: chartPayload,
    });
    router.push(`/advisor?session=${sessionId}`);
  };

  const activeAccount = useMemo(() => {
    return accounts.find((a) => a.account_id === selectedAccountId) || null;
  }, [selectedAccountId, accounts]);

  const activeAccountName = activeAccount?.bank_name ? `${activeAccount.bank_name} (${activeAccount.currency || "GBP"})` : "Select Account";

  const dropdownItems = useMemo(() => {
    return accounts.map(a => ({ ...a, id: a.account_id }));
  }, [accounts]);

  const isActuallyLoading = isInitialLoading || accounts.length === 0;

  return (
    <WidgetFlipCard
      insight={undefined}
      isLoading={false}
      isDataLoading={isInitialLoading}
      onDiscuss={handleDiscuss}
    >
      <Card className="w-full h-full">
        <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
          <div className="flex justify-between items-start w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
              Expenditure & Forecast
            </h3>
            <div className="flex items-center gap-1">
              <FlipButton />
              <CloseButton
                onPress={onRemove}
                className="w-8 h-8 min-w-8 opacity-50 hover:opacity-100 hover:bg-white/10 text-foreground transition-all rounded-full"
              />
            </div>
          </div>

          <div className="flex flex-wrap items-end gap-4 w-full pointer-events-auto">
            <DatePicker
              className="flex-1 min-w-35 max-w-45"
              name="From Date"
              value={startDate}
              onChange={setStartDate}
            >
              <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 mb-2 block pl-1">
                From Date
              </Label>
              <DateField.Group
                fullWidth
                className="bg-white/5 border-[0.5px] border-white/10 rounded-xl px-4 h-12 flex items-center transition-all focus-within:border-primary/50 shadow-inner"
              >
                <DateField.Input className="flex-1  text-foreground text-[11px] font-mono ">
                  {(segment) => (
                    <DateField.Segment
                      segment={segment}
                      className="focus:bg-primary/20 rounded-md px-1 outline-none"
                    />
                  )}
                </DateField.Input>
                <DateField.Suffix className="ml-2 flex items-center">
                  <DatePicker.Trigger className="text-foreground/30 hover:text-primary cursor-pointer transition-colors">
                    <CalendarIcon size={14} />
                  </DatePicker.Trigger>
                </DateField.Suffix>
              </DateField.Group>
              <DatePicker.Popover className="popover min-w-max p-6">
                <Calendar aria-label="From date" className="w-full min-w-65">
                  <Calendar.Header className="flex items-center gap-3 mb-6">
                    <Calendar.YearPickerTrigger className="flex items-center gap-2 mr-auto cursor-pointer hover:opacity-70 transition-opacity">
                      <Calendar.YearPickerTriggerHeading className="text-sm font-black uppercase tracking-widest text-primary italic" />
                      <Calendar.YearPickerTriggerIndicator className="text-foreground/30 w-4 h-4" />
                    </Calendar.YearPickerTrigger>
                    <Calendar.NavButton
                      slot="previous"
                      className="text-primary w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/5 cursor-pointer transition-colors"
                    />
                    <Calendar.NavButton
                      slot="next"
                      className="text-primary w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/5 cursor-pointer transition-colors"
                    />
                  </Calendar.Header>
                  <Calendar.Grid className="w-full border-collapse">
                    <Calendar.GridHeader>
                      {(day) => (
                        <Calendar.HeaderCell className="text-[9px] font-black text-foreground/20 pb-4 text-center uppercase tracking-widest">
                          {day}
                        </Calendar.HeaderCell>
                      )}
                    </Calendar.GridHeader>
                    <Calendar.GridBody>
                      {(date) => (
                        <Calendar.Cell
                          date={date}
                          className="w-8 h-8 flex items-center justify-center mx-auto text-[11px] font-mono text-foreground rounded-lg hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                        />
                      )}
                    </Calendar.GridBody>
                  </Calendar.Grid>
                  <Calendar.YearPickerGrid>
                    <Calendar.YearPickerGridBody>
                      {({year}) => (
                        <Calendar.YearPickerCell
                          year={year}
                          className="h-8 px-2 w-full flex items-center justify-center mx-auto text-xs font-mono text-foreground rounded-lg hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                        />
                      )}
                    </Calendar.YearPickerGridBody>
                  </Calendar.YearPickerGrid>
                </Calendar>
              </DatePicker.Popover>
            </DatePicker>

            <DatePicker
              className="flex-1 min-w-35 max-w-45"
              name="To Date"
              value={endDate}
              onChange={setEndDate}
            >
              <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 mb-2 block pl-1">
                To Date
              </Label>
              <DateField.Group
                fullWidth
                className="bg-white/5 border-[0.5px] border-white/10 rounded-xl px-4 h-12 flex items-center transition-all focus-within:border-primary/50 shadow-inner"
              >
                <DateField.Input className="flex-1  text-foreground text-[11px] font-mono ">
                  {(segment) => (
                    <DateField.Segment
                      segment={segment}
                      className="focus:bg-primary/20 rounded-md px-1 outline-none"
                    />
                  )}
                </DateField.Input>
                <DateField.Suffix className="ml-2 flex items-center">
                  <DatePicker.Trigger className="text-foreground/30 hover:text-primary cursor-pointer transition-colors">
                    <CalendarIcon size={14} />
                  </DatePicker.Trigger>
                </DateField.Suffix>
              </DateField.Group>
              <DatePicker.Popover className="popover min-w-max p-6">
                <Calendar aria-label="To date" className="w-full min-w-65">
                  <Calendar.Header className="flex items-center gap-3 mb-6">
                    <Calendar.YearPickerTrigger className="flex items-center gap-2 mr-auto cursor-pointer hover:opacity-70 transition-opacity">
                      <Calendar.YearPickerTriggerHeading className="text-sm font-black uppercase tracking-widest text-primary italic" />
                      <Calendar.YearPickerTriggerIndicator className="text-foreground/30 w-4 h-4" />
                    </Calendar.YearPickerTrigger>
                    <Calendar.NavButton
                      slot="previous"
                      className="text-primary w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/5 cursor-pointer transition-colors"
                    />
                    <Calendar.NavButton
                      slot="next"
                      className="text-primary w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/5 cursor-pointer transition-colors"
                    />
                  </Calendar.Header>
                  <Calendar.Grid className="w-full border-collapse">
                    <Calendar.GridHeader>
                      {(day) => (
                        <Calendar.HeaderCell className="text-[9px] font-black text-foreground/20 pb-4 text-center uppercase tracking-widest">
                          {day}
                        </Calendar.HeaderCell>
                      )}
                    </Calendar.GridHeader>
                    <Calendar.GridBody>
                      {(date) => (
                        <Calendar.Cell
                          date={date}
                          className="w-8 h-8 flex items-center justify-center mx-auto text-[11px] font-mono text-foreground rounded-lg hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                        />
                      )}
                    </Calendar.GridBody>
                  </Calendar.Grid>
                  <Calendar.YearPickerGrid>
                    <Calendar.YearPickerGridBody>
                      {({year}) => (
                        <Calendar.YearPickerCell
                          year={year}
                          className="h-8 px-2 w-full flex items-center justify-center mx-auto text-xs font-mono text-foreground rounded-lg hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-all"
                        />
                      )}
                    </Calendar.YearPickerGridBody>
                  </Calendar.YearPickerGrid>
                </Calendar>
              </DatePicker.Popover>
            </DatePicker>

            <div className="flex-1 min-w-40 max-w-50 ml-auto">
              <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 mb-2 block pl-1">
                Selected Account
              </Label>
              <Dropdown>
                <Dropdown.Trigger className="h-12 min-h-12 w-full bg-white/5 hover:bg-white/10 border-[0.5px] border-white/10 text-[10px] text-foreground font-black uppercase tracking-widest rounded-xl px-4 flex items-center justify-between transition-all cursor-pointer outline-none focus:border-primary/50 shadow-inner">
                  <span className="truncate pointer-events-none">
                    {activeAccountName}
                  </span>
                  <ChevronDown
                    size={14}
                    className="text-foreground/30 shrink-0 pointer-events-none"
                  />
                </Dropdown.Trigger>
                <Dropdown.Popover className="bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-xl shadow-2xl w-64 p-2">
                  <Dropdown.Menu
                    items={accounts.map(a => ({ ...a, id: a.account_id }))}
                    className="outline-none"
                    selectedKeys={new Set([selectedAccountId])}
                    onSelectionChange={(keys: any) => {
                      const val = Array.from(keys)[0] as string;
                      if (val) setSelectedAccountId(val);
                    }}
                    selectionMode="single"
                  >
                    {(acc: any) => (
                      <Dropdown.Item
                        key={acc.id}
                        id={acc.id}
                        textValue={acc.bank_name}
                        className="flex flex-col px-4 py-3 rounded-lg hover:bg-white/10 cursor-pointer outline-none transition-all"
                      >
                        <div className="w-full relative flex items-center justify-between">
                            <span className="text-[11px] font-black text-foreground uppercase tracking-tight pr-4 italic truncate">
                              {acc.bank_name} ({acc.currency || "GBP"})
                            </span>
                            {selectedAccountId === acc.id && (
                              <div className="bg-primary border-none w-1.5 h-1.5 min-w-0 p-0 relative transform-none rounded-full shrink-0 shadow-[0_0_10px_rgba(0,242,255,0.6)]" />
                            )}
                          </div>
                        <span className="text-foreground/20 text-[9px] font-mono tracking-widest mt-1.5 uppercase">
                          Account No: *{acc.account_number?.slice(-4)}
                        </span>
                      </Dropdown.Item>
                    )}
                  </Dropdown.Menu>
                </Dropdown.Popover>
              </Dropdown>
            </div>
          </div>

          <div className="flex flex-col gap-3 w-full mt-2">
            <div className="flex items-center gap-3 mb-1 px-1">
              <CalendarIcon size={14} className="text-primary/60" />
              <span className="text-[9px] font-black uppercase tracking-[0.4em] text-foreground/30">
                Time types
              </span>
            </div>
            <ToggleButtonGroup
              disallowEmptySelection
              className="bg-white/5 backdrop-blur-md p-1 rounded-xl w-64 flex flex-row gap-1 border-[0.5px] border-white/5 shadow-inner"
              selectedKeys={new Set([granularity])}
              selectionMode="single"
              size="sm"
              onSelectionChange={(keys) => {
                const first = Array.from(keys)[0];
                if (first) setGranularity(first as string);
              }}
            >
              {[
                { id: "daily", label: "Daily" },
                { id: "weekly", label: "Weekly" },
                { id: "monthly", label: "Monthly" },
              ].map((g) => (
                <ToggleButton
                  key={g.id}
                  id={g.id}
                  variant="ghost"
                  className={cn(
                    "flex-1 py-2 rounded-lg text-[9px] font-black uppercase tracking-widest transition-all h-auto border-none",
                    granularity === g.id
                      ? "bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 shadow-lg"
                      : "text-foreground/40 hover:text-foreground data-[hovered=true]:bg-white/10",
                  )}
                >
                  {g.label}
                </ToggleButton>
              ))}
            </ToggleButtonGroup>
          </div>
        </Card.Header>

        <Card.Content className="flex-1 w-full min-h-0 flex flex-col p-0 z-0 bg-white/1 border-t-[0.5px] border-white/5 relative overflow-hidden">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,242,255,0.02)_0%,transparent_70%)] pointer-events-none" />
          <div className="mb-2 shrink-0 p-8">
            {isActuallyLoading ? (
              <div className="space-y-3">
                <Skeleton
                  animationType="shimmer"
                  className="h-10 w-40 rounded-xl bg-white/5"
                />
                <Skeleton
                  animationType="shimmer"
                  className="h-3 w-56 rounded-lg bg-white/5"
                />
              </div>
            ) : (
              <>
                <div className="flex items-baseline gap-4 w-full justify-between pr-8">
                  <div className="flex flex-col">
                    <div className="flex items-baseline gap-4">
                      <h4 className="text-4xl font-normal text-foreground tracking-tighter mb-1 font-mono">
                        {new Intl.NumberFormat("en-GB", {
                          style: "currency",
                          currency: activeAccount?.currency || "GBP",
                          minimumFractionDigits: 2,
                          maximumFractionDigits: 2,
                        }).format(totalExpenses)}
                      </h4>
                      {isFetching && (
                        <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                      )}
                    </div>
                    <p className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.3em]">
                      Total Expenditure
                    </p>
                  </div>
                  <Button
                    isIconOnly
                    size="sm"
                    variant="ghost"
                    className="w-8 h-8 min-w-8 bg-white/5 text-foreground/60 hover:text-primary rounded-lg border-[0.5px] border-white/10"
                    onPress={() => setIsModalOpen(true)}
                  >
                    <Settings size={14} />
                  </Button>
                </div>
              </>
            )}
          </div>

          <div className="flex-1 w-full min-h-62.5 mb-6 relative flex items-center justify-center px-8">
            {(isActuallyLoading || !config) ? (
              <Skeleton
                className="w-full h-full rounded-xl bg-white/5"
                animationType="shimmer"
              />
            ) : (
              <div className="w-full h-full relative">
                {isFetching && (
                  <div className="absolute top-0 left-0 w-full z-30 pointer-events-none flex items-center">
                    <ProgressBar isIndeterminate aria-label="Loading..." size="sm" color="accent" className="w-full" />
                    <div className="absolute top-2 right-6">
                      <Spinner size="sm" color="accent" />
                    </div>
                  </div>
                )}
                <CoreChartEngine config={config} />
              </div>
            )}
          </div>
        </Card.Content>
      </Card>
      <SimulationControlsModal
        isOpen={isModalOpen}
        onClose={() => setIsModalOpen(false)}
        onApply={setSimulationOverrides}
        initialValues={simulationOverrides}
      />
    </WidgetFlipCard>
  );
}
