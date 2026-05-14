// app/(protected)/_components/ExpenseDistributionWidget.tsx
import React, { useState, useEffect, useMemo } from "react";
import { ChevronDown, PieChart } from "lucide-react";
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

interface ExpenseDistributionWidgetProps {
  onRemove?: () => void;
}

const NEON_COLORS = [
  "#00E5FF",
  "#FF007F",
  "#7FFF00",
  "#B900FF",
  "#FFEA00",
  "#FF3366",
  "#00F0FF",
  "#39FF14",
];

export default function ExpenseDistributionWidget({
  onRemove,
}: ExpenseDistributionWidgetProps) {
  const { accounts } = useBudAI();

  const [localAccountId, setLocalAccountId] = useState<string>(
    accounts.length > 0 ? accounts[0].account_id : "",
  );

  const [startDate, setStartDate] = useState<DateValue | null>(
    today(getLocalTimeZone()).subtract({ months: 1 }),
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

    const fromStr = `${startDate.year}-${String(startDate.month).padStart(2, "0")}-${String(startDate.day).padStart(2, "0")}`;
    const toStr = `${endDate.year}-${String(endDate.month).padStart(2, "0")}-${String(endDate.day).padStart(2, "0")}`;

    const bankName =
      accounts.find((a) => a.account_id === localAccountId)?.bank_name ||
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
        bank_name_or_id: localAccountId,
        from_date: fromStr,
        to_date: toStr,
      },
      "Spending Distribution",
    );
  }, [aggregatedData, startDate, endDate, localAccountId, accounts]);

  const activeAccountName =
    accounts.find((a) => a.account_id === localAccountId)?.bank_name ||
    "Select Account";

  return (
    <Card className="w-full h-full obsidian-glass p-6 flex flex-col shadow-2xl relative overflow-hidden">
      <Card.Header className="flex flex-col gap-4 mb-6 p-0 shrink-0 w-full z-10">
        <div className="flex justify-between items-start w-full">
          <Card.Title className="text-white font-bold text-2xl tracking-tight">
            Spending Distribution
          </Card.Title>
          {onRemove && (
            <CloseButton
              onPress={onRemove}
              className="text-[#8B8E98] hover:bg-white/10 hover:text-white transition-colors"
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
              className="bg-[#181A20] border border-[#2A2D35] rounded-xl px-3 h-10 flex items-center transition-colors focus-within:border-cyan-600"
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
            <DatePicker.Popover className="bg-black border-2 border-cyan-400/70 rounded-2xl p-4 shadow-2xl z-50">
              <Calendar aria-label="From date" className="w-full min-w-65">
                <Calendar.Header className="flex items-center gap-2 mb-4">
                  <Calendar.YearPickerTrigger className="flex items-center gap-1 mr-auto cursor-pointer hover:opacity-80 transition-opacity">
                    <Calendar.YearPickerTriggerHeading className="text-base font-semibold text-cyan-400" />
                    <Calendar.YearPickerTriggerIndicator className="text-[#8B8E98] w-4 h-4" />
                  </Calendar.YearPickerTrigger>
                  <Calendar.NavButton
                    slot="previous"
                    className="text-cyan-400 w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/10 cursor-pointer transition-colors"
                  />
                  <Calendar.NavButton
                    slot="next"
                    className="text-cyan-400 w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/10 cursor-pointer transition-colors"
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
                        className="w-8 h-8 flex items-center justify-center mx-auto text-sm text-white rounded-full hover:bg-white/10 data-[selected=true]:bg-cyan-600 data-[selected=true]:text-white cursor-pointer outline-none transition-colors"
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
              className="bg-[#181A20] border border-[#2A2D35] rounded-xl px-3 h-10 flex items-center transition-colors focus-within:border-cyan-600"
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
            <DatePicker.Popover className="bg-black border-2 border-cyan-400/70 rounded-2xl p-4 shadow-2xl z-50">
              <Calendar aria-label="To date" className="w-full min-w-65">
                <Calendar.Header className="flex items-center gap-2 mb-4">
                  <Calendar.YearPickerTrigger className="flex items-center gap-1 mr-auto cursor-pointer hover:opacity-80 transition-opacity">
                    <Calendar.YearPickerTriggerHeading className="text-base font-semibold text-cyan-400" />
                    <Calendar.YearPickerTriggerIndicator className="text-[#8B8E98] w-4 h-4" />
                  </Calendar.YearPickerTrigger>
                  <Calendar.NavButton
                    slot="previous"
                    className="text-cyan-400 w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/10 cursor-pointer transition-colors"
                  />
                  <Calendar.NavButton
                    slot="next"
                    className="text-cyan-400 w-8 h-8 flex items-center justify-center rounded-lg hover:bg-white/10 cursor-pointer transition-colors"
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
                        className="w-8 h-8 flex items-center justify-center mx-auto text-sm text-white rounded-full hover:bg-white/10 data-[selected=true]:bg-cyan-600 data-[selected=true]:text-white cursor-pointer outline-none transition-colors"
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
                  className="h-10 min-h-10 min-w-30 max-w-50 bg-[#181A20] hover:bg-white/10 border border-[#2A2D35] text-sm text-white font-medium rounded-xl px-4 flex items-center justify-between transition-all cursor-pointer outline-none"
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
              <Dropdown.Popover className="bg-black border-2 border-cyan-400/70 shadow-2xl rounded-2xl min-w-50 z-50">
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
                      className="rounded-xl transition-colors data-[hover=true]:bg-white/5 py-3 px-3 outline-none cursor-pointer"
                    >
                      <div className="flex flex-col gap-1 w-full">
                        <Badge.Anchor className="w-full relative flex items-center justify-between">
                          <Label className="text-sm font-semibold text-white cursor-pointer pointer-events-none pr-4">
                            {acc.bank_name}
                          </Label>
                          {localAccountId === acc.account_id && (
                            <Badge className="bg-green-500 border-none w-2.5 h-2.5 min-w-0 p-0 relative transform-none rounded-full shrink-0" />
                          )}
                        </Badge.Anchor>
                        <Description className="text-xs text-[#8B8E98] font-mono tracking-wider">
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

      <Card.Content className="flex-1 w-full min-h-0 p-0 z-0 overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
        <div className="flex flex-col min-h-full">
          <div className="mb-2 shrink-0 px-2">
            <h4 className="text-3xl font-bold text-white tracking-tight mb-1">
              £{" "}
              {totalExpenses.toLocaleString(undefined, {
                minimumFractionDigits: 2,
                maximumFractionDigits: 2,
              })}
            </h4>
            <p className="text-xs font-medium text-[#5E6272]">
              Total expenses for selected period
            </p>
          </div>

          <div className="flex-1 w-full min-h-62.5 mb-6 relative flex items-center justify-center shrink-0">
            {isLoading ? (
              <Spinner color="accent" />
            ) : config ? (
              <CoreChartEngine config={config} />
            ) : (
              <div className="flex flex-col items-center justify-center h-full w-full opacity-50">
                <div className="w-12 h-12 rounded-full border-2 border-dashed border-[#5E6272] flex items-center justify-center mb-3">
                  <PieChart size={20} className="text-[#5E6272]" />
                </div>
                <span className="text-[#5E6272] text-sm font-medium">
                  No Data Available
                </span>
              </div>
            )}
          </div>

          <div className="space-y-4 shrink-0 px-2 pb-4">
            {aggregatedData.map((c, idx) => (
              <div key={idx} className="flex justify-between items-center">
                <div className="flex items-center gap-3">
                  <div
                    className="w-3 h-3 rounded-md"
                    style={{
                      backgroundColor: NEON_COLORS[idx % NEON_COLORS.length],
                    }}
                  ></div>
                  <span className="text-sm text-[#8B8E98] font-medium">
                    {c.name}
                  </span>
                </div>
                <span className="text-sm font-semibold text-white">
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
  );
}
