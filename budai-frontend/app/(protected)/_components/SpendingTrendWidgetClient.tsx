"use client";

import React, { useState, useMemo } from "react";
import { ChevronDown, Activity, Calendar as CalendarIcon } from "lucide-react";
import CoreChartEngine from "./CoreChartEngine";
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
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { today, getLocalTimeZone, DateValue } from "@internationalized/date";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { useAdvisorInsight, useSpendingTrends, usePersistedState } from "@/lib/hooks";
import WidgetFlipCard from "./WidgetFlipCard";
import { useRouter } from "next/navigation";
import { Account, BankChartData } from "@/types";
import { cn } from "@/lib/utils";
import { WidgetContext } from "../home/DashboardClient";

interface SpendingTrendWidgetProps {
  initialData?: BankChartData[];
}

export default function SpendingTrendWidgetClient({
  initialData,
}: SpendingTrendWidgetProps) {
  const router = useRouter();
  const { onRemove } = React.useContext(WidgetContext);
  const { accounts, createNewSession } = useBudAI();

  const [selectedAccountId, setSelectedAccountId] = usePersistedState<string>(
    "spending_trend_account",
    accounts[0]?.account_id || "",
  );
  const [granularity, setGranularity] = useState<string>("monthly");

  const [startDate, setStartDate] = useState<DateValue | null>(
    today(getLocalTimeZone()).subtract({ months: 6 }),
  );

  const [endDate, setEndDate] = useState<DateValue | null>(
    today(getLocalTimeZone()),
  );

  const fromStr = startDate
    ? `${startDate.year}-${String(startDate.month).padStart(2, "0")}-${String(startDate.day).padStart(2, "0")}`
    : "";
  const toStr = endDate
    ? `${endDate.year}-${String(endDate.month).padStart(2, "0")}-${String(endDate.day).padStart(2, "0")}`
    : "";

  const {
    data: chartPayload = [],
    isLoading: isInitialLoading,
    isFetching,
  } = useSpendingTrends(
    selectedAccountId,
    fromStr,
    toStr,
    granularity as "daily" | "weekly" | "monthly",
    granularity === "monthly" ? initialData : undefined,
  );

  const totalExpenses = useMemo(() => {
    let total = 0;
    chartPayload.forEach((bank: BankChartData) => {
      bank.data.forEach((point: Record<string, string | number>) => {
        total += Number(point.Amount || point.amount || 0);
      });
    });
    return total;
  }, [chartPayload]);

  const config = useMemo(() => {
    if (!startDate || !endDate || chartPayload.length === 0) return null;

    const chartType =
      granularity === "daily"
        ? "historical_daily"
        : granularity === "weekly"
          ? "historical_weekly"
          : "historical_monthly";

    return buildChartConfig(
      chartType,
      chartPayload,
      {
        bank_name_or_id: selectedAccountId,
        from_date: fromStr,
        to_date: toStr,
      },
      `${granularity.charAt(0).toUpperCase() + granularity.slice(1)} Spending Trends`,
      { disableAnimation: true },
    );
  }, [
    chartPayload,
    startDate,
    endDate,
    selectedAccountId,
    fromStr,
    toStr,
    granularity,
  ]);

  const { data: insight, isLoading: isAnalyzing } = useAdvisorInsight(
    "spendingTrend",
    chartPayload,
  );

  const handleDiscuss = () => {
    const sessionId = createNewSession("Spending Trend Deep Dive", {
      type: "spending_trend",
      accountId: selectedAccountId,
      data: chartPayload,
    });
    router.push(`/advisor?session=${sessionId}`);
  };

  const activeAccountName = useMemo(() => {
    return (
      accounts.find((a) => a.account_id === selectedAccountId)?.bank_name ||
      "Select Account"
    );
  }, [selectedAccountId, accounts]);

  const dropdownItems = useMemo((): Account[] => {
    return [...accounts];
  }, [accounts]);

  const isActuallyLoading = isInitialLoading || accounts.length === 0;

  return (
    <WidgetFlipCard
      insight={insight}
      isLoading={isAnalyzing}
      isDataLoading={isInitialLoading}
      onDiscuss={handleDiscuss}
    >
      <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
        <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
          <div className="flex justify-between items-start w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
              Past Expenditure
            </h3>
            <CloseButton
              onPress={onRemove}
              className="text-foreground/20 hover:text-foreground transition-all rounded-md"
            />
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
                <DateField.Input className="flex-1 bg-transparent text-foreground text-[11px] font-mono outline-none">
                  {(segment) => (
                    <DateField.Segment
                      segment={segment}
                      className="focus:bg-primary/20 rounded-md px-1 outline-none"
                    />
                  )}
                </DateField.Input>
                <DateField.Suffix className="ml-2 flex items-center">
                  <DatePicker.Trigger className="text-foreground/30 hover:text-primary cursor-pointer transition-colors">
                    <DatePicker.TriggerIndicator />
                  </DatePicker.Trigger>
                </DateField.Suffix>
              </DateField.Group>
              <DatePicker.Popover className="bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-2xl p-6 shadow-2xl z-50">
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
                <DateField.Input className="flex-1 bg-transparent text-foreground text-[11px] font-mono outline-none">
                  {(segment) => (
                    <DateField.Segment
                      segment={segment}
                      className="focus:bg-primary/20 rounded-md px-1 outline-none"
                    />
                  )}
                </DateField.Input>
                <DateField.Suffix className="ml-2 flex items-center">
                  <DatePicker.Trigger className="text-foreground/30 hover:text-primary cursor-pointer transition-colors">
                    <DatePicker.TriggerIndicator />
                  </DatePicker.Trigger>
                </DateField.Suffix>
              </DateField.Group>
              <DatePicker.Popover className="bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-2xl p-6 shadow-2xl z-50">
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
                </Calendar>
              </DatePicker.Popover>
            </DatePicker>

            <div className="ml-auto flex shrink-0">
              <Dropdown>
                <Dropdown.Trigger className="h-12 min-h-12 min-w-40 max-w-55 bg-white/5 hover:bg-white/10 border-[0.5px] border-white/10 text-[11px] text-foreground font-black uppercase tracking-widest rounded-xl px-5 flex items-center justify-between transition-all cursor-pointer outline-none focus:border-primary/50 shadow-inner">
                  <span className="truncate pointer-events-none">
                    {activeAccountName}
                  </span>
                  <ChevronDown
                    size={14}
                    className="text-foreground/30 shrink-0 pointer-events-none"
                  />
                </Dropdown.Trigger>
                <Dropdown.Popover
                  className="bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 shadow-2xl rounded-2xl min-w-50 z-50"
                  placement="bottom left"
                >
                  <Dropdown.Menu
                    items={dropdownItems}
                    selectionMode="single"
                    selectedKeys={new Set([selectedAccountId])}
                    onSelectionChange={(keys: Selection) => {
                      if (keys !== "all") {
                        const selectedValue = Array.from(keys)[0];
                        if (selectedValue)
                          setSelectedAccountId(String(selectedValue));
                      }
                    }}
                    className="p-2"
                  >
                    {(acc: Account) => (
                      <Dropdown.Item
                        id={acc.account_id}
                        textValue={acc.bank_name}
                        className="rounded-xl transition-all data-[hover=true]:bg-white/10 py-3 px-4 outline-none cursor-pointer w-full block border-[0.5px] border-transparent data-[hover=true]:border-primary/30"
                      >
                        <div className="flex flex-col w-full">
                          <Badge.Anchor className="w-full relative flex items-center justify-between">
                            <Label className="text-[11px] font-black text-foreground uppercase tracking-tight cursor-pointer pointer-events-none pr-4 italic">
                              {acc.bank_name}
                            </Label>
                            {selectedAccountId === acc.account_id && (
                              <Badge className="bg-primary border-none w-1.5 h-1.5 min-w-0 p-0 relative transform-none rounded-full shrink-0 shadow-[0_0_10px_rgba(0,127,255,0.6)]" />
                            )}
                          </Badge.Anchor>
                          <Description className="text-[9px] text-foreground/30 font-mono tracking-[0.2em] pointer-events-none mt-1.5 uppercase">
                            Account No: *{acc.account_number?.slice(-4) || "0000"}
                          </Description>
                        </div>
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
                      ? "bg-primary text-primary-foreground shadow-lg"
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
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,127,255,0.02)_0%,transparent_70%)] pointer-events-none" />
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
                <div className="flex items-baseline gap-4">
                  <h4 className="text-4xl font-normal text-foreground tracking-tighter mb-1 font-mono">
                    £
                    {totalExpenses.toLocaleString(undefined, {
                      minimumFractionDigits: 2,
                      maximumFractionDigits: 2,
                    })}
                  </h4>
                  {isFetching && (
                    <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                  )}
                </div>
                <p className="text-[9px] font-black text-foreground/30 uppercase tracking-[0.3em]">
                  Total Outflow Vector
                </p>
              </>
            )}
          </div>

          <div className="flex-1 w-full min-h-62.5 mb-6 relative flex items-center justify-center px-8">
            {isActuallyLoading && chartPayload.length === 0 ? (
              <Skeleton
                className="w-full h-full rounded-xl bg-white/5"
                animationType="shimmer"
              />
            ) : config ? (
              <div className="w-full h-full">
                <CoreChartEngine config={config} />
              </div>
            ) : (
              <div className="flex flex-col items-center justify-center h-full w-full opacity-20">
                <div className="w-12 h-12 rounded-full border-[0.5px] border-white/40 flex items-center justify-center mb-4">
                  <Activity size={24} className="text-foreground" />
                </div>
                <span className="text-[10px] font-black uppercase tracking-[0.4em]">
                  Awaiting Logic Samples
                </span>
              </div>
            )}
          </div>
        </Card.Content>
      </Card>
    </WidgetFlipCard>
  );
}
