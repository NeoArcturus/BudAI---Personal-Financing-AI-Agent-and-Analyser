// app/(protected)/_components/CashFlowWidget.tsx
import React, { useState, useEffect, useMemo } from "react";
import { ChevronDown, BarChart2 } from "lucide-react";
import CoreChartEngine from "./CoreChartEngine";
import type { Selection } from "@heroui/react";
import {
  Card,
  Dropdown,
  Label,
  Description,
  Spinner,
  DatePicker,
  DateField,
  Calendar,
  CloseButton,
  Badge,
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import { Transaction, Account, BankChartData } from "@/types";
import { today, getLocalTimeZone, DateValue } from "@internationalized/date";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";

interface CashFlowWidgetProps {
  onRemove?: () => void;
}

export default function CashFlowWidget({ onRemove }: CashFlowWidgetProps) {
  const { accounts } = useBudAI();

  const [localAccountId, setLocalAccountId] = useState<string>(
    accounts.length > 0 ? accounts[0].account_id : "",
  );

  const [startDate, setStartDate] = useState<DateValue | null>(
    today(getLocalTimeZone()).subtract({ years: 1 }),
  );

  const [endDate, setEndDate] = useState<DateValue | null>(
    today(getLocalTimeZone()),
  );

  const [transactions, setTransactions] = useState<Transaction[]>([]);
  const [isLoading, setIsLoading] = useState<boolean>(false);

  useEffect(() => {
    if (!localAccountId && accounts.length > 0) {
      setLocalAccountId(accounts[0].account_id);
    }
  }, [accounts, localAccountId]);

  useEffect(() => {
    let isMounted = true;
    const fetchWidgetData = async () => {
      if (!localAccountId || !startDate || !endDate) return;
      setIsLoading(true);
      try {
        const fromStr = `${startDate.year}-${String(startDate.month).padStart(2, "0")}-${String(startDate.day).padStart(2, "0")}`;
        const toStr = `${endDate.year}-${String(endDate.month).padStart(2, "0")}-${String(endDate.day).padStart(2, "0")}`;
        const queryStr = `?from=${fromStr}&to=${toStr}`;

        const response = await apiFetch(
          `/api/accounts/${localAccountId}/transactions${queryStr}`,
          {},
          true,
        );
        if (response.ok) {
          const data = await response.json();
          if (isMounted) {
            setTransactions((data.transactions as Transaction[]) || []);
          }
        }
      } catch (error) {
        console.error(error);
      } finally {
        if (isMounted) setIsLoading(false);
      }
    };
    fetchWidgetData();
    return () => {
      isMounted = false;
    };
  }, [localAccountId, startDate, endDate]);

  const config = useMemo(() => {
    if (!startDate || !endDate) return null;

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
      accounts.find((a) => a.account_id === localAccountId)?.bank_name ||
      "Account";

    const payload: BankChartData[] = [
      {
        bank_name: bankName,
        data: chartData,
      },
    ];

    const fromStr = `${startDate.year}-${String(startDate.month).padStart(2, "0")}-${String(startDate.day).padStart(2, "0")}`;
    const toStr = `${endDate.year}-${String(endDate.month).padStart(2, "0")}-${String(endDate.day).padStart(2, "0")}`;

    return buildChartConfig(
      "cash_flow_mixed",
      payload,
      {
        bank_name_or_id: localAccountId,
        from_date: fromStr,
        to_date: toStr,
      },
      "Money Flow",
    );
  }, [transactions, startDate, endDate, localAccountId, accounts]);

  const activeAccountName =
    accounts.find((a) => a.account_id === localAccountId)?.bank_name ||
    "Select Account";

  return (
    <Card className="w-full h-full obsidian-glass p-6 flex flex-col shadow-2xl relative overflow-hidden font-geist">
      <Card.Header className="flex flex-col gap-4 mb-6 p-0 shrink-0 w-full z-10">
        <div className="flex justify-between items-start w-full">
          <Card.Title className="text-white font-bold text-2xl tracking-tight">
            Money Flow
          </Card.Title>
          {onRemove && (
            <CloseButton
              onPress={onRemove}
              className="text-[#8B8E98] hover:bg-white/10 hover:text-white transition-colors rounded-2xl"
            />
          )}
        </div>

        <div className="flex flex-wrap items-end gap-3 w-full pointer-events-auto">
          <DatePicker
            className="flex-1 min-w-35 max-w-45"
            name="From Date"
            value={startDate}
            onChange={setStartDate}
          >
            <Label className="text-white text-xs font-medium mb-1.5 block opacity-80">
              From Date
            </Label>
            <DateField.Group
              fullWidth
              className="bg-[#181A20] border border-white/8 rounded-xl px-3 h-10 flex items-center transition-colors focus-within:border-neon-cyan"
            >
              <DateField.Input className="flex-1 bg-transparent text-white text-sm outline-none">
                {(segment) => (
                  <DateField.Segment
                    segment={segment}
                    className="focus:bg-white/20 rounded px-0.5 outline-none"
                  />
                )}
              </DateField.Input>
              <DateField.Suffix className="ml-2 flex items-center">
                <DatePicker.Trigger className="text-[#8B8E98] hover:text-white cursor-pointer transition-colors">
                  <DatePicker.TriggerIndicator />
                </DatePicker.Trigger>
              </DateField.Suffix>
            </DateField.Group>
            <DatePicker.Popover className="bg-obsidian border border-neon-cyan/50 rounded-2xl p-4 shadow-2xl z-50 backdrop-blur-xl">
              <Calendar aria-label="From date" className="w-full min-w-65">
                <Calendar.Header className="flex items-center gap-2 mb-4">
                  <Calendar.YearPickerTrigger className="flex items-center gap-1 mr-auto cursor-pointer hover:opacity-80 transition-opacity">
                    <Calendar.YearPickerTriggerHeading className="text-base font-semibold text-neon-cyan" />
                    <Calendar.YearPickerTriggerIndicator className="text-[#8B8E98] w-4 h-4" />
                  </Calendar.YearPickerTrigger>
                  <Calendar.NavButton
                    slot="previous"
                    className="text-neon-cyan w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/10 cursor-pointer transition-colors"
                  />
                  <Calendar.NavButton
                    slot="next"
                    className="text-neon-cyan w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/10 cursor-pointer transition-colors"
                  />
                </Calendar.Header>
                <Calendar.Grid className="w-full border-collapse">
                  <Calendar.GridHeader>
                    {(day) => (
                      <Calendar.HeaderCell className="text-xs font-medium text-[#8B8E98] pb-3 text-center">
                        {day}
                      </Calendar.HeaderCell>
                    )}
                  </Calendar.GridHeader>
                  <Calendar.GridBody>
                    {(date) => (
                      <Calendar.Cell
                        date={date}
                        className="w-8 h-8 flex items-center justify-center mx-auto text-sm text-white rounded-full hover:bg-white/10 data-[selected=true]:bg-neon-cyan data-[selected=true]:text-obsidian cursor-pointer outline-none transition-colors"
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
            <Label className="text-white text-xs font-medium mb-1.5 block opacity-80">
              To Date
            </Label>
            <DateField.Group
              fullWidth
              className="bg-[#181A20] border border-white/8 rounded-xl px-3 h-10 flex items-center transition-colors focus-within:border-neon-cyan"
            >
              <DateField.Input className="flex-1 bg-transparent text-white text-sm outline-none">
                {(segment) => (
                  <DateField.Segment
                    segment={segment}
                    className="focus:bg-white/20 rounded px-0.5 outline-none"
                  />
                )}
              </DateField.Input>
              <DateField.Suffix className="ml-2 flex items-center">
                <DatePicker.Trigger className="text-[#8B8E98] hover:text-white cursor-pointer transition-colors">
                  <DatePicker.TriggerIndicator />
                </DatePicker.Trigger>
              </DateField.Suffix>
            </DateField.Group>
            <DatePicker.Popover className="bg-obsidian border border-neon-cyan/50 rounded-2xl p-4 shadow-2xl z-50 backdrop-blur-xl">
              <Calendar aria-label="To date" className="w-full min-w-65">
                <Calendar.Header className="flex items-center gap-2 mb-4">
                  <Calendar.YearPickerTrigger className="flex items-center gap-1 mr-auto cursor-pointer hover:opacity-80 transition-opacity">
                    <Calendar.YearPickerTriggerHeading className="text-base font-semibold text-neon-cyan" />
                    <Calendar.YearPickerTriggerIndicator className="text-[#8B8E98] w-4 h-4" />
                  </Calendar.YearPickerTrigger>
                  <Calendar.NavButton
                    slot="previous"
                    className="text-neon-cyan w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/10 cursor-pointer transition-colors"
                  />
                  <Calendar.NavButton
                    slot="next"
                    className="text-neon-cyan w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/10 cursor-pointer transition-colors"
                  />
                </Calendar.Header>
                <Calendar.Grid className="w-full border-collapse">
                  <Calendar.GridHeader>
                    {(day) => (
                      <Calendar.HeaderCell className="text-xs font-medium text-[#8B8E98] pb-3 text-center">
                        {day}
                      </Calendar.HeaderCell>
                    )}
                  </Calendar.GridHeader>
                  <Calendar.GridBody>
                    {(date) => (
                      <Calendar.Cell
                        date={date}
                        className="w-8 h-8 flex items-center justify-center mx-auto text-sm text-white rounded-full hover:bg-white/10 data-[selected=true]:bg-neon-cyan data-[selected=true]:text-obsidian cursor-pointer outline-none transition-colors"
                      />
                    )}
                  </Calendar.GridBody>
                </Calendar.Grid>
              </Calendar>
            </DatePicker.Popover>
          </DatePicker>

          <div className="ml-auto flex shrink-0">
            <Dropdown>
              <Dropdown.Trigger>
                <div
                  role="button"
                  tabIndex={0}
                  className="h-10 min-h-10 min-w-30 max-w-50 bg-[#181A20] hover:bg-white/10 border border-white/8 text-sm text-white font-medium rounded-xl px-4 flex items-center justify-between transition-all cursor-pointer outline-none focus-within:border-neon-cyan"
                >
                  <span className="truncate pointer-events-none">
                    {activeAccountName}
                  </span>
                  <ChevronDown
                    size={16}
                    className="text-[#8B8E98] shrink-0 pointer-events-none"
                  />
                </div>
              </Dropdown.Trigger>
              <Dropdown.Popover className="bg-obsidian border border-neon-cyan/50 shadow-2xl rounded-2xl min-w-40 z-50 backdrop-blur-xl">
                <Dropdown.Menu
                  items={accounts}
                  selectionMode="single"
                  selectedKeys={new Set([localAccountId])}
                  onSelectionChange={(keys: Selection) => {
                    if (keys !== "all") {
                      const selectedValue = Array.from(keys)[0];
                      if (selectedValue)
                        setLocalAccountId(String(selectedValue));
                    }
                  }}
                  className="p-2"
                >
                  {(acc: Account) => (
                    <Dropdown.Item
                      id={acc.account_id}
                      textValue={acc.bank_name}
                      className="rounded-xl transition-all data-[hover=true]:bg-white/5 data-[hover=true]:ring-1 data-[hover=true]:ring-neon-cyan/70 border border-transparent data-[hover=true]:border-neon-cyan/70 py-3 px-3 outline-none cursor-pointer focus:ring-0 focus:outline-none w-full block"
                    >
                      <div className="flex flex-col w-full">
                        <Badge.Anchor className="w-full relative flex items-center justify-between">
                          <Label className="text-sm font-semibold text-white cursor-pointer pointer-events-none pr-4">
                            {acc.bank_name}
                          </Label>
                          {localAccountId === acc.account_id && (
                            <Badge className="bg-brand-green border-none w-2.5 h-2.5 min-w-0 p-0 relative transform-none rounded-full shrink-0" />
                          )}
                        </Badge.Anchor>
                        <Description className="text-xs text-[#8B8E98] font-mono tracking-wider pointer-events-none mt-1">
                          •••• {acc.account_number?.slice(-4) || "0000"}
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

      <Card.Content className="flex-1 w-full min-h-0 flex items-center justify-center p-0 z-0 [&::-webkit-scrollbar]:hidden">
        {isLoading ? (
          <Spinner color="accent" />
        ) : config ? (
          <CoreChartEngine config={config} />
        ) : (
          <div className="flex flex-col items-center justify-center h-full w-full opacity-50">
            <div className="w-12 h-12 rounded-full border-2 border-dashed border-[#5E6272] flex items-center justify-center mb-3">
              <BarChart2 size={20} className="text-[#5E6272]" />
            </div>
            <span className="text-[#5E6272] text-sm font-medium">
              No Data Available
            </span>
          </div>
        )}
      </Card.Content>
    </Card>
  );
}
