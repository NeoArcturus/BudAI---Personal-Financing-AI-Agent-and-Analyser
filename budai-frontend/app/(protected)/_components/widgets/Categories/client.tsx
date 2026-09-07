"use client";

import React, { useState, useMemo } from "react";
import { ChevronDown, PieChart, Calendar as CalendarIcon } from "lucide-react";
import CoreChartEngine from "../../internal/ChartEngine";
import type { Selection } from "@heroui/react";
import {
  Card,
  ListBox,
  Dropdown,
  Label,
  Description,
  Skeleton,
  DatePicker,
  DateField,
  Calendar,
  Badge,
  CloseButton,
  Spinner,
  ProgressBar,
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { Account, BankChartData, Transaction } from "@/types";
import { today, getLocalTimeZone, DateValue } from "@internationalized/date";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { useExpenseCategories, usePersistedState, usePersistedDate } from "@/lib/hooks";
import WidgetFlipCard, { FlipButton } from "../../internal/FlipCard";
import { useRouter } from "next/navigation";
import { WidgetContext } from "../../../home/DashboardClient";

interface ExpenseDistributionWidgetProps {
  initialData?: Transaction[];
}

const NEON_COLORS = [
  "#00F2FF",
  "#A855F7",
  "#EC4899",
  "#22C55E",
  "#6366F1",
  "#14B8A6",
  "#F43F5E",
  "#8B5CF6",
];

export default function ExpenseDistributionWidgetClient({
  initialData,
}: ExpenseDistributionWidgetProps) {
  const router = useRouter();
  const { onRemove, instanceId } = React.useContext(WidgetContext);
  const { accounts, createNewSession } = useBudAI();

  const [selectedAccountId, setSelectedAccountId] = usePersistedState<string>(
    `expense_dist_account${instanceId ? `-${instanceId}` : ""}`,
    accounts[0]?.account_id || "",
  );

  React.useEffect(() => {
    if (accounts.length > 0) {
      if (!selectedAccountId || selectedAccountId.startsWith("react-aria-") || !accounts.find(a => a.account_id === selectedAccountId)) {
        setSelectedAccountId(accounts[0].account_id);
      }
    }
  }, [accounts, selectedAccountId, setSelectedAccountId]);

  const [startDate, setStartDate] = usePersistedDate(`cat_start_${instanceId || ""}`,
    today(getLocalTimeZone()).subtract({ months: 3 }),
  );

  const [endDate, setEndDate] = usePersistedDate(`cat_end_${instanceId || ""}`,
    today(getLocalTimeZone()),
  );

  const fromStr = startDate
    ? `${startDate.year}-${String(startDate.month).padStart(2, "0")}-${String(startDate.day).padStart(2, "0")}`
    : "";
  const toStr = endDate
    ? `${endDate.year}-${String(endDate.month).padStart(2, "0")}-${String(endDate.day).padStart(2, "0")}`
    : "";

  const {
    data: chartData = [],
    isLoading: isInitialLoading,
    isFetching,
  } = useExpenseCategories(selectedAccountId, fromStr, toStr, true);

  const aggregatedData = useMemo(() => {
    if (!chartData || chartData.length === 0 || !chartData[0].data) return [];

    return chartData[0].data
      .map((item: any) => ({
        name: item.Category || item.category || item.Date || item.date || "Other",
        value: item.Total_Amount || item.total_amount || item.Amount || item.amount || 0,
      }))
      .filter((item) => item.value > 0)
      .sort((a, b) => b.value - a.value);
  }, [chartData]);

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

  const activeAccount = useMemo(() => {
    return accounts.find((a) => a.account_id === selectedAccountId) || null;
  }, [selectedAccountId, accounts]);

  const activeAccountName = activeAccount?.bank_name ? `${activeAccount.bank_name} (${activeAccount.currency || "GBP"})` : "Select Account";

  return (
    <WidgetFlipCard
      insight={undefined}
      isLoading={false}
      onDiscuss={handleDiscuss}
    >
      <Card className=" border-none h-full">
        <Card.Header className="flex flex-col gap-4 p-8 pb-4 shrink-0 w-full z-10">
          <div className="flex justify-between items-start w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0 flex items-center gap-2">
              Expense Breakdown
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
                      {({ year }) => (
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
                      {({ year }) => (
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

            <div className="flex-1 min-w-40 max-w-50">
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
                    onSelectionChange={(keys: Selection) => {
                      const val = Array.from(keys)[0] as string;
                      setSelectedAccountId(val);
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
        </Card.Header>

        <Card.Content className="flex-1 w-full min-h-0 p-6 pt-2 z-0 overflow-hidden pointer-events-auto">
          <div className="flex flex-row items-center w-full h-full gap-6">
            <div className="w-[55%] h-full relative flex items-center justify-center shrink-0">
              {(isInitialLoading || !config) ? (
                <Skeleton
                  animationType="shimmer"
                  className="w-48 h-48 rounded-full bg-white/5"
                />
              ) : (
                <div
                  className={`w-full h-full relative flex items-center justify-center ${isFetching ? "opacity-70 transition-opacity" : ""}`}
                >
                  {isFetching && (
                    <div className="absolute top-0 left-0 w-full z-30 pointer-events-none flex items-center">
                      <ProgressBar isIndeterminate aria-label="Loading..." size="sm" color="accent" className="w-full" />
                    </div>
                  )}
                  <CoreChartEngine config={{ ...config, options: { ...config.options, cutout: "65%", plugins: { ...config.options?.plugins, legend: { display: false } } } as any }} />

                  {/* Center Text absolute overlay */}
                  <div className="absolute inset-0 flex flex-col items-center justify-center pointer-events-none z-10">
                    <span className="text-[20px] font-mono tracking-tighter text-foregrond font-bold">
                      {new Intl.NumberFormat("en-GB", {
                        style: "currency",
                        currency: activeAccount?.currency || "GBP",
                        minimumFractionDigits: 0,
                        maximumFractionDigits: 0,
                      }).format(totalExpenses)}
                    </span>
                    <span className="text-[8px] uppercase tracking-[0.2em] font-black text-muted-foreground mt-1">
                      Total Expenses
                    </span>
                  </div>
                </div>
              )}
            </div>

            <div className="w-[45%] h-full overflow-y-auto shrink-0 pb-4 pr-2 scrollbar-hide">
              {(isInitialLoading || aggregatedData.length === 0)
                ? (
                  <div className="flex flex-col gap-3">
                    {Array.from({ length: 5 }).map((_, i) => (
                      <div key={i} className="flex justify-between items-center w-full p-2 bg-white/5 rounded-xl border border-white/5">
                        <div className="flex items-center gap-3">
                          <Skeleton className="w-3 h-3 rounded-full bg-white/10" />
                          <Skeleton className="h-3 w-20 rounded bg-white/10" />
                        </div>
                        <Skeleton className="h-3 w-12 rounded bg-white/10" />
                      </div>
                    ))}
                  </div>
                )
                : (
                  <ListBox
                    aria-label="Expense Categories"
                    selectionMode="none"
                    className="w-full p-0 gap-3"
                  >
                    {aggregatedData.map((c: any, idx: number) => (
                      <ListBox.Item
                        key={c.name}
                        textValue={c.name}
                        id={c.name}
                        className="w-full p-3 bg-white/5 hover:bg-white/10 data-[hover=true]:bg-white/10 rounded-xl border border-white/5 transition-colors cursor-pointer"
                      >
                        <div className="flex justify-between items-center w-full">
                          <div className="flex items-center gap-3 truncate max-w-[50%]">
                            <div
                              className="w-2.5 h-2.5 rounded-full shrink-0 shadow-sm"
                              style={{
                                backgroundColor: NEON_COLORS[idx % NEON_COLORS.length],
                                boxShadow: `0 0 8px ${NEON_COLORS[idx % NEON_COLORS.length]}80`
                              }}
                            />
                            <span className="text-xs font-bold tracking-wide text-foreground truncate">
                              {c.name}
                            </span>
                          </div>
                          <div className="flex items-center gap-3 shrink-0">
                            <span className="font-mono text-sm tracking-tighter text-red-400 font-bold">
                              {new Intl.NumberFormat("en-GB", {
                                style: "currency",
                                currency: activeAccount?.currency || "GBP",
                                minimumFractionDigits: 0,
                                maximumFractionDigits: 0,
                              }).format(c.value)}
                            </span>
                            <span className="text-[9px] font-mono font-black text-muted-foreground w-8 text-right bg-black/40 px-2 py-1 rounded-md flex items-center justify-center">
                              {((c.value / (totalExpenses || 1)) * 100).toFixed(1)}%
                            </span>
                          </div>
                        </div>
                      </ListBox.Item>
                    ))}
                  </ListBox>
                )}
            </div>
          </div>
        </Card.Content>
      </Card>
    </WidgetFlipCard>
  );
}
