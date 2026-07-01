"use client";

import React, { useState, useMemo } from "react";
import { ChevronDown, BarChart2, Calendar as CalendarIcon } from "lucide-react";
import CoreChartEngine from "./CoreChartEngine";
import type { Selection } from "@heroui/react";
import {
  Card,
  Dropdown,
  Label,
  Skeleton,
  DatePicker,
  DateField,
  Calendar,
  CloseButton,
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { Transaction, BankChartData } from "@/types";
import { today, getLocalTimeZone, DateValue } from "@internationalized/date";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { useTransactions, usePersistedState } from "@/lib/hooks";
import WidgetFlipCard from "./WidgetFlipCard";
import { useRouter } from "next/navigation";
import { WidgetContext } from "../home/DashboardClient";

interface CashFlowWidgetProps {
  initialData?: Transaction[];
}

export default function CashFlowWidgetClient({
  initialData,
}: CashFlowWidgetProps) {
  const router = useRouter();
  const { onRemove } = React.useContext(WidgetContext);
  const { accounts, createNewSession } = useBudAI();

  const [selectedAccountId, setSelectedAccountId] = usePersistedState<string>(
    "cashflow_account",
    accounts[0]?.account_id || "",
  );

  const [startDate, setStartDate] = useState<DateValue | null>(
    today(getLocalTimeZone()).subtract({ years: 1 }),
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

  const { data: transactions = [], isLoading } = useTransactions(
    selectedAccountId,
    fromStr,
    toStr,
    initialData,
  );

  const chartConfig = useMemo(() => {
    if (!startDate || !endDate || transactions.length === 0) return null;

    const getMonthsInRange = (start: DateValue, end: DateValue) => {
      const months = [];
      let currYear = start.year;
      let currMonth = start.month;

      while (
        currYear < end.year ||
        (currYear === end.year && currMonth <= end.month)
      ) {
        const date = new Date(currYear, currMonth - 1, 1);
        const monthYear = date.toLocaleDateString("en-US", {
          month: "short",
          year: "2-digit",
        });
        months.push(monthYear);

        currMonth++;
        if (currMonth > 12) {
          currMonth = 1;
          currYear++;
        }
      }
      return months;
    };

    const allMonths = getMonthsInRange(startDate, endDate);
    const monthlyData: Record<string, { Income: number; Expense: number }> = {};

    allMonths.forEach((m) => {
      monthlyData[m] = { Income: 0, Expense: 0 };
    });

    transactions.forEach((tx) => {
      const date = new Date(tx.timestamp || tx.date || "");
      if (isNaN(date.getTime())) return;

      const monthYear = date.toLocaleDateString("en-US", {
        month: "short",
        year: "2-digit",
      });

      if (monthlyData[monthYear]) {
        const amount = tx.amount || tx.Amount || 0;
        if (
          amount > 0 ||
          tx.category?.toLowerCase() === "income" ||
          tx.Category?.toLowerCase() === "income"
        ) {
          monthlyData[monthYear].Income += Math.abs(amount);
        } else {
          monthlyData[monthYear].Expense += Math.abs(amount);
        }
      }
    });

    const chartData = allMonths.map((m) => ({
      Month: m,
      Income: monthlyData[m].Income,
      Expense: monthlyData[m].Expense,
      Net_Balance: monthlyData[m].Income - monthlyData[m].Expense,
    }));

    const bankName =
      accounts.find((a) => a.account_id === selectedAccountId)?.bank_name ||
      "Account";

    const payload: BankChartData[] = [
      {
        bank_name: bankName,
        data: chartData,
      },
    ];

    return buildChartConfig(
      "cash_flow_mixed",
      payload,
      {
        bank_name_or_id: selectedAccountId,
        from_date: fromStr,
        to_date: toStr,
      },
      "Liquidity Intelligence",
    );
  }, [
    transactions,
    selectedAccountId,
    fromStr,
    toStr,
    startDate,
    endDate,
    accounts,
  ]);

  const handleDiscuss = () => {
    const sessionId = createNewSession("Cash Flow Analysis", {
      type: "cash_flow",
      accountId: selectedAccountId,
      data: transactions,
    });
    router.push(`/advisor?session=${sessionId}`);
  };

  const selectedAccountName = useMemo(() => {
    const acc = accounts.find((a) => a.account_id === selectedAccountId);
    return acc ? acc.bank_name : "Select Account";
  }, [selectedAccountId, accounts]);

  return (
    <WidgetFlipCard
      insight={undefined}
      isLoading={false}
      onDiscuss={handleDiscuss}
    >
      <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
        <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
          <div className="flex justify-between items-start w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
              Cashflow Insights
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
                    <CalendarIcon size={14} />
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
                    <CalendarIcon size={14} />
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

            <div className="flex-1 min-w-40 max-w-50">
              <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 mb-2 block pl-1">
                Selected Account
              </Label>
              <Dropdown>
                <Dropdown.Trigger className="h-12 min-h-12 w-full bg-white/5 hover:bg-white/10 border-[0.5px] border-white/10 text-[10px] text-foreground font-black uppercase tracking-widest rounded-xl px-4 flex items-center justify-between transition-all cursor-pointer outline-none focus:border-primary/50 shadow-inner">
                  <span className="truncate pointer-events-none">
                    {selectedAccountName}
                  </span>
                  <ChevronDown
                    size={14}
                    className="text-foreground/30 shrink-0 pointer-events-none"
                  />
                </Dropdown.Trigger>
                <Dropdown.Popover className="bg-black/80 backdrop-blur-3xl border-[0.5px] border-white/10 rounded-xl shadow-2xl w-64 z-50 p-2">
                  <Dropdown.Menu
                    className="outline-none"
                    selectedKeys={new Set([selectedAccountId])}
                    onSelectionChange={(keys: Selection) => {
                      const val = Array.from(keys)[0] as string;
                      setSelectedAccountId(val);
                    }}
                    selectionMode="single"
                  >
                    {accounts.map((acc) => (
                      <Dropdown.Item
                        key={acc.account_id}
                        textValue={acc.bank_name}
                        className="flex flex-col px-4 py-3 rounded-lg hover:bg-white/10 cursor-pointer outline-none transition-all"
                      >
                        <span className="text-foreground font-black text-[11px] uppercase tracking-tight italic">
                          {acc.bank_name}
                        </span>
                        <span className="text-foreground/20 text-[9px] font-mono tracking-widest mt-1.5 uppercase">
                          Account No: *{acc.account_number?.slice(-4)}
                        </span>
                      </Dropdown.Item>
                    ))}
                  </Dropdown.Menu>
                </Dropdown.Popover>
              </Dropdown>
            </div>
          </div>
        </Card.Header>

        <Card.Content className="flex-1 w-full flex items-center justify-center p-8 pt-0 relative overflow-hidden">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,242,255,0.02)_0%,transparent_70%)] pointer-events-none" />
          {isLoading ? (
            <Skeleton
              animationType="shimmer"
              className="w-full h-full rounded-2xl bg-white/5"
            />
          ) : chartConfig ? (
            <div className="w-full h-full relative z-10">
              <CoreChartEngine config={chartConfig} />
            </div>
          ) : (
            <div className="text-foreground/10 text-center flex flex-col items-center gap-4">
              <div className="w-16 h-16 rounded-full border-[0.5px] border-white/5 flex items-center justify-center bg-white/1">
                <BarChart2 size={32} />
              </div>
              <span className="text-[10px] font-black uppercase tracking-[0.4em]">
                Awaiting Data Streams
              </span>
            </div>
          )}
        </Card.Content>
      </Card>
    </WidgetFlipCard>
  );
}
