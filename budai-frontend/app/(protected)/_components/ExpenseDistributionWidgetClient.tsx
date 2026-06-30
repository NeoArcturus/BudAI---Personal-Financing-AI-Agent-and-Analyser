"use client";

import React, { useState, useMemo } from "react";
import { ChevronDown, PieChart, Calendar as CalendarIcon } from "lucide-react";
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
  CloseButton,
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { Account, BankChartData, Transaction } from "@/types";
import { today, getLocalTimeZone, DateValue } from "@internationalized/date";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { useTransactions, usePersistedState } from "@/lib/hooks";
import WidgetFlipCard from "./WidgetFlipCard";
import { useRouter } from "next/navigation";
import { WidgetContext } from "../home/DashboardClient";

interface ExpenseDistributionWidgetProps {
  initialData?: Transaction[];
}

const NEON_COLORS = [
  "#007FFF",
  "#3B82F6",
  "#60A5FA",
  "#93C5FD",
  "#BFDBFE",
  "#0F52BA",
  "#1E40AF",
  "#1D4ED8",
];

export default function ExpenseDistributionWidgetClient({
  initialData,
}: ExpenseDistributionWidgetProps) {
  const router = useRouter();
  const { onRemove } = React.useContext(WidgetContext);
  const { accounts, createNewSession } = useBudAI();

  const [selectedAccountId, setSelectedAccountId] = usePersistedState<string>(
    "expense_dist_account",
    accounts[0]?.account_id || "",
  );

  const [startDate, setStartDate] = useState<DateValue | null>(
    today(getLocalTimeZone()).subtract({ months: 3 }),
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
    data: transactions = [],
    isLoading: isInitialLoading,
    isFetching,
  } = useTransactions(selectedAccountId, fromStr, toStr, initialData);

  const aggregatedData = useMemo(() => {
    const categories: Record<string, number> = {};

    transactions.forEach((tx) => {
      const amount = tx.amount || tx.Amount || 0;
      if (
        amount > 0 &&
        tx.category?.toLowerCase() !== "expense" &&
        tx.Category?.toLowerCase() !== "expense"
      ) {
        return;
      }

      const cat = tx.category || tx.Category || "Other";
      if (!categories[cat]) categories[cat] = 0;
      categories[cat] += Math.abs(amount);
    });

    return Object.entries(categories)
      .map(([name, value]) => ({ name, value }))
      .sort((a, b) => b.value - a.value);
  }, [transactions]);

  const totalExpenses = useMemo(() => {
    return aggregatedData.reduce((acc, curr) => acc + curr.value, 0);
  }, [aggregatedData]);

  const config = useMemo(() => {
    if (!startDate || !endDate || aggregatedData.length === 0) return null;

    const bankName =
      accounts.find((a) => a.account_id === selectedAccountId)?.bank_name ||
      "Account";

    const payload: BankChartData[] = [
      {
        bank_name: bankName,
        data: aggregatedData.map((d) => ({
          Category: d.name,
          Total_Amount: d.value,
        })),
      },
    ];

    return buildChartConfig(
      "categorized_doughnut",
      payload,
      {
        bank_name_or_id: selectedAccountId,
        from_date: fromStr,
        to_date: toStr,
      },
      "Spending Distribution",
    );
  }, [
    aggregatedData,
    startDate,
    endDate,
    selectedAccountId,
    accounts,
    fromStr,
    toStr,
  ]);

  const handleDiscuss = () => {
    const sessionId = createNewSession("Expense Distribution Analysis", {
      type: "expense_distribution",
      accountId: selectedAccountId,
      data: aggregatedData,
    });
    router.push(`/advisor?session=${sessionId}`);
  };

  const activeAccountName = useMemo(() => {
    return (
      accounts.find((a) => a.account_id === selectedAccountId)?.bank_name ||
      "Select Account"
    );
  }, [selectedAccountId, accounts]);

  return (
    <WidgetFlipCard
      insight={undefined}
      isLoading={false}
      onDiscuss={handleDiscuss}
    >
      <Card className="liquid-glass border-none h-full rounded-xl flex flex-col relative overflow-hidden">
        <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
          <div className="flex justify-between items-start w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
              Expense Category Distribution
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
                To End
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
                    items={accounts}
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
                              <Badge className="bg-primary border-none w-1.5 h-1.5 min-w-0 p-0 relative transform-none rounded-full shrink-0 shadow-[0_0_10px_rgba(0,242,255,0.6)]" />
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
        </Card.Header>

        <Card.Content className="flex-1 w-full min-h-0 p-0 z-0 overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
          <div className="flex flex-col min-h-full">
            <div className="mb-2 shrink-0 px-2">
              {isInitialLoading ? (
                <div className="space-y-2">
                  <Skeleton
                    animationType="shimmer"
                    className="h-9 w-40 rounded-xl bg-secondary"
                  />
                  <Skeleton
                    animationType="shimmer"
                    className="h-4 w-56 rounded-lg bg-secondary"
                  />
                </div>
              ) : (
                <>
                  <div className="flex items-baseline gap-2">
                    <h4 className="text-3xl font-bold text-foreground tracking-tight mb-1">
                      £{" "}
                      {totalExpenses.toLocaleString(undefined, {
                        minimumFractionDigits: 2,
                        maximumFractionDigits: 2,
                      })}
                    </h4>
                    {isFetching && (
                      <div className="w-2 h-2 rounded-full bg-primary animate-pulse" />
                    )}
                  </div>
                  <p className="text-xs font-medium text-muted-foreground">
                    Total expenses for selected period
                  </p>
                </>
              )}
            </div>

            <div className="flex-1 w-full min-h-62.5 mb-6 relative flex items-center justify-center shrink-0">
              {isInitialLoading && transactions.length === 0 ? (
                <Skeleton
                  animationType="shimmer"
                  className="w-48 h-48 rounded-full bg-secondary"
                />
              ) : config ? (
                <div
                  className={`w-full h-full flex items-center justify-center ${isFetching ? "opacity-70 transition-opacity" : ""}`}
                >
                  <CoreChartEngine config={config} />
                </div>
              ) : (
                <div className="flex flex-col items-center justify-center h-full w-full opacity-50">
                  <div className="w-12 h-12 rounded-full border-2 border-dashed border-muted-foreground flex items-center justify-center mb-3">
                    <PieChart size={20} className="text-muted-foreground" />
                  </div>
                  <span className="text-muted-foreground text-sm font-medium">
                    No Data Available
                  </span>
                </div>
              )}
            </div>

            <div className="space-y-4 shrink-0 px-2 pb-4">
              {isInitialLoading && transactions.length === 0
                ? Array.from({ length: 4 }).map((_, i) => (
                    <div key={i} className="flex justify-between items-center">
                      <div className="flex items-center gap-3">
                        <Skeleton
                          animationType="shimmer"
                          className="w-3 h-3 rounded-md bg-secondary"
                        />
                        <Skeleton
                          animationType="shimmer"
                          className="h-4 w-24 rounded bg-secondary"
                        />
                      </div>
                      <Skeleton
                        animationType="shimmer"
                        className="h-4 w-16 rounded bg-secondary"
                      />
                    </div>
                  ))
                : aggregatedData.map((c, idx) => (
                    <div
                      key={idx}
                      className={`flex justify-between items-center ${isFetching ? "opacity-50" : ""}`}
                    >
                      <div className="flex items-center gap-3">
                        <div
                          className="w-3 h-3 rounded-md"
                          style={{
                            backgroundColor:
                              NEON_COLORS[idx % NEON_COLORS.length],
                          }}
                        ></div>
                        <span className="text-sm text-muted-foreground font-medium">
                          {c.name}
                        </span>
                      </div>
                      <span className="text-sm font-semibold text-foreground">
                        £{" "}
                        {c.value.toLocaleString(undefined, {
                          minimumFractionDigits: 0,
                          maximumFractionDigits: 0,
                        })}
                      </span>
                    </div>
                  ))}
            </div>
          </div>
        </Card.Content>
      </Card>
    </WidgetFlipCard>
  );
}
