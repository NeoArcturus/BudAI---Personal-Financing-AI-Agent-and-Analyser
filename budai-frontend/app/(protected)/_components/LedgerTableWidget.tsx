// app/(protected)/_components/LedgerTableWidget.tsx
import React, { useState, useEffect, useCallback } from "react";
import { ChevronDown, ListVideo, Pencil } from "lucide-react";
import { Transaction, Account } from "@/types";
import type { Selection } from "@heroui/react";
import {
  Button,
  Card,
  Table,
  Dropdown,
  Label,
  Description,
  Chip,
  DatePicker,
  DateField,
  Calendar,
  Modal,
  Avatar,
  Badge,
  CloseButton,
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";
import { parseDate, CalendarDate } from "@internationalized/date";

interface LedgerTableWidgetProps {
  onRemove?: () => void;
}

// Strictly type the potential fields coming from various banking APIs (TrueLayer, Plaid, etc.)
type ExtendedTx = Transaction & {
  transaction_uuid?: string;
  id?: string;
  transaction_id?: string;
  TransactionId?: string;
  category?: string;
  Category?: string;
  amount?: number;
  Amount?: number;
  description?: string;
  Description?: string;
  timestamp?: string;
  date?: string;
};

const STANDARD_CATEGORIES = [
  "Food & Dining",
  "Transportation",
  "Bills & Utilities",
  "Shopping",
  "Entertainment",
  "Health & Wellness",
  "Transfers & Investments",
  "High-Risk / Anomaly",
  "Income",
];

export default function LedgerTableWidget({
  onRemove,
}: LedgerTableWidgetProps) {
  const { accounts } = useBudAI();
  const [localAccountId, setLocalAccountId] = useState<string>(
    accounts.length > 0 ? accounts[0].account_id : "",
  );
  const [transactions, setTransactions] = useState<ExtendedTx[]>([]);
  const [fromDate, setFromDate] = useState<CalendarDate | null>(
    parseDate("2025-11-10"),
  );
  const [toDate, setToDate] = useState<CalendarDate | null>(
    parseDate("2026-05-10"),
  );
  const [isLoading, setIsLoading] = useState<boolean>(false);

  const [selectedTx, setSelectedTx] = useState<ExtendedTx | null>(null);
  const [editCategory, setEditCategory] = useState<string>("");
  const [isUpdating, setIsUpdating] = useState<boolean>(false);

  useEffect(() => {
    if (!localAccountId && accounts.length > 0) {
      setLocalAccountId(accounts[0].account_id);
    }
  }, [accounts, localAccountId]);

  const fetchTransactions = useCallback(
    async (fromStr?: string, toStr?: string) => {
      if (!localAccountId) return;
      setIsLoading(true);
      try {
        const queryParams = new URLSearchParams();
        if (fromStr) queryParams.append("from", fromStr);
        if (toStr) queryParams.append("to", toStr);
        const queryStr = queryParams.toString()
          ? `?${queryParams.toString()}`
          : "";

        const response = await apiFetch(
          `/api/accounts/${localAccountId}/transactions${queryStr}`,
          {},
          true,
        );
        if (response.ok) {
          const data = await response.json();
          setTransactions((data.transactions as ExtendedTx[]) || []);
        }
      } catch (error) {
        console.error(error);
      } finally {
        setIsLoading(false);
      }
    },
    [localAccountId],
  );

  useEffect(() => {
    fetchTransactions(fromDate?.toString(), toDate?.toString());
  }, [fetchTransactions, fromDate, toDate]);

  const handleUpdateCategory = async () => {
    if (!selectedTx || !editCategory) return;

    // Safely extract the ID regardless of the banking API format
    const txId =
      selectedTx.transaction_uuid ||
      selectedTx.id ||
      selectedTx.transaction_id ||
      selectedTx.TransactionId;

    if (!txId) {
      console.error("Missing required transaction ID for update.", {
        availableFields: Object.keys(selectedTx),
      });
      return;
    }

    setIsUpdating(true);
    try {
      const res = await apiFetch(
        "/api/categorizer/labels",
        {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            transaction_uuid: String(txId),
            corrected_label: editCategory,
            retrain_model: true,
          }),
        },
        true, // Ensure JWT is passed
      );

      if (res.ok) {
        setTransactions((prev) =>
          prev.map((t) => {
            const currentId =
              t.transaction_uuid || t.id || t.transaction_id || t.TransactionId;
            return currentId === txId
              ? { ...t, category: editCategory, Category: editCategory }
              : t;
          }),
        );
        setSelectedTx(null);
      } else {
        const errorData = await res.json().catch(() => ({}));
        console.error(
          "Backend failed to update category:",
          res.status,
          errorData,
        );
      }
    } catch (e) {
      console.error("Network error during category update:", e);
    } finally {
      setIsUpdating(false);
    }
  };

  const activeAccountName =
    accounts.find((a) => a.account_id === localAccountId)?.bank_name ||
    "Select Account";

  const getCategoryTheme = (category: string) => {
    const themes: Record<string, string> = {
      "Food & Dining":
        "bg-[#FF007F]/20 text-[#FF007F] border-[#FF007F]/40 shadow-[0_0_10px_rgba(255,0,127,0.15)]",
      Transportation:
        "bg-[#FFEA00]/20 text-[#FFEA00] border-[#FFEA00]/40 shadow-[0_0_10px_rgba(255,234,0,0.15)]",
      "Bills & Utilities":
        "bg-neon-cyan/20 text-neon-cyan border-neon-cyan/40 shadow-[0_0_10px_rgba(0,240,255,0.15)]",
      Shopping:
        "bg-[#B900FF]/20 text-[#B900FF] border-[#B900FF]/40 shadow-[0_0_10px_rgba(185,0,255,0.15)]",
      Entertainment:
        "bg-deep-pink/20 text-deep-pink border-deep-pink/40 shadow-[0_0_10px_rgba(255,51,102,0.15)]",
      "Health & Wellness":
        "bg-neon-cyan/20 text-neon-cyan border-neon-cyan/40 shadow-[0_0_10px_rgba(0,229,255,0.15)]",
      "Transfers & Investments":
        "bg-[#7FFF00]/20 text-[#7FFF00] border-[#7FFF00]/40 shadow-[0_0_10px_rgba(127,255,0,0.15)]",
      "High-Risk / Anomaly":
        "bg-[#FF0000]/20 text-[#FF0000] border-[#FF0000]/40 shadow-[0_0_10px_rgba(255,0,0,0.15)]",
      Income:
        "bg-[#39FF14]/20 text-[#39FF14] border-[#39FF14]/40 shadow-[0_0_10px_rgba(57,255,20,0.15)]",
    };
    return (
      themes[category] || "bg-[#8B8E98]/20 text-[#8B8E98] border-[#8B8E98]/40"
    );
  };

  const formatShortDate = (dateStr: string) => {
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return dateStr;
    const day = d.toLocaleDateString("en-US", { weekday: "short" });
    const time = d
      .toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })
      .toLowerCase();
    return `${day} ${time}`;
  };

  return (
    <Card className="w-full h-full bg-obsidian/40 backdrop-blur-xl rounded-3xl border border-white/8 shadow-2xl flex flex-col overflow-hidden relative font-geist">
      <Card.Header className="flex flex-col p-6 border-b border-white/8 shrink-0 gap-4 w-full">
        <div className="flex justify-between items-center w-full">
          <Card.Title className="text-white font-bold text-2xl tracking-tight whitespace-nowrap">
            Transaction history
          </Card.Title>
          {onRemove && (
            <CloseButton
              onPress={onRemove}
              className="text-[#8B8E98] hover:bg-white/10 hover:text-white transition-colors rounded-2xl"
            />
          )}
        </div>

        <div className="flex flex-wrap sm:flex-nowrap items-center justify-start gap-2 w-full pointer-events-auto">
          <DatePicker
            className="w-50 sm:min-w-35 sm:max-w-40"
            name="From Date"
            value={fromDate}
            onChange={setFromDate}
          >
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

          <span className="text-[#5E6272]">-</span>

          <DatePicker
            className="w-50 sm:min-w-35 sm:max-w-40"
            name="To Date"
            value={toDate}
            onChange={setToDate}
          >
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
              <Dropdown.Popover className="bg-obsidian border border-neon-cyan/50 shadow-2xl rounded-2xl min-w-50 z-50 backdrop-blur-xl">
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

      <Card.Content className="flex-1 h-full w-full p-0 overflow-hidden">
        {transactions.length === 0 ? (
          <div className="flex flex-col items-center justify-center h-full w-full opacity-50">
            <div className="w-12 h-12 rounded-full border-2 border-dashed border-[#5E6272] flex items-center justify-center mb-3">
              <ListVideo size={20} className="text-[#5E6272]" />
            </div>
            <span className="text-[#5E6272] text-sm font-medium">
              {isLoading ? "Loading transactions..." : "No transactions found."}
            </span>
          </div>
        ) : (
          <Table className="w-full h-full text-left relative table-fixed">
            <Table.ScrollContainer className="h-full w-full overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] px-2 pb-4">
              <Table.Content aria-label="Transaction Ledger" className="w-full">
                <Table.Header className="sticky top-0 bg-obsidian/90 backdrop-blur-md z-10 border-b border-white/8 w-full">
                  <Table.Column
                    isRowHeader
                    className="py-4 px-6 font-semibold uppercase tracking-wider text-[10px] text-[#5E6272] w-[20%] sm:w-[40%]"
                  >
                    Transaction
                  </Table.Column>
                  <Table.Column className="py-4 px-6 font-semibold uppercase tracking-wider text-[10px] text-[#5E6272] w-[30%] sm:w-[15%]">
                    Amount
                  </Table.Column>
                  <Table.Column className="py-4 px-6 font-semibold uppercase tracking-wider text-[10px] text-[#5E6272] w-[25%] sm:w-[15%]">
                    Date
                  </Table.Column>
                  <Table.Column className="py-4 px-6 font-semibold uppercase tracking-wider text-[10px] text-[#5E6272] w-[25%] sm:w-[20%]">
                    Category
                  </Table.Column>
                  <Table.Column className="py-4 px-6 font-semibold uppercase tracking-wider text-[10px] text-[#5E6272] hidden sm:table-cell sm:w-[10%] text-center">
                    Actions
                  </Table.Column>
                </Table.Header>
                <Table.Body className="w-full">
                  {transactions.map((tx, i) => {
                    const desc = tx.description || tx.Description || "Unknown";
                    const amount = tx.amount ?? tx.Amount ?? 0;
                    const cat = tx.category || tx.Category || "Uncategorized";
                    const initial = desc.charAt(0).toUpperCase();
                    const isPositive = amount > 0 || cat === "Income";

                    return (
                      <Table.Row
                        key={i}
                        className="hover:bg-white/5 transition-colors group border-b border-white/5 last:border-0 w-full cursor-pointer sm:cursor-default"
                        onClick={() => {
                          if (window.innerWidth < 640) {
                            setSelectedTx(tx);
                            setEditCategory(cat);
                          }
                        }}
                      >
                        <Table.Cell className="py-4 px-6 w-[20%] sm:w-[40%]">
                          <div className="flex items-center gap-4 w-full">
                            <Avatar
                              variant="soft"
                              color="accent"
                              className={
                                isPositive
                                  ? "w-9 h-9 shrink-0 shadow-md rounded-full flex items-center justify-center border-2 border-neon-cyan bg-neon-cyan/40"
                                  : "w-9 h-9 shrink-0 shadow-md rounded-full flex items-center justify-center border-2 border-deep-pink bg-deep-pink/40"
                              }
                            >
                              <Avatar.Fallback className="w-full h-full flex items-center justify-center font-bold text-sm leading-none m-0 p-0 text-white">
                                {initial}
                              </Avatar.Fallback>
                            </Avatar>
                            <span className="hidden sm:block text-white text-sm font-medium truncate w-full group-hover:text-white/90 transition-colors">
                              {desc}
                            </span>
                          </div>
                        </Table.Cell>
                        <Table.Cell className="py-4 px-6 w-[30%] sm:w-[15%]">
                          <span
                            className={cn(
                              "text-sm font-medium",
                              isPositive ? "text-neon-cyan" : "text-deep-pink",
                            )}
                          >
                            {isPositive ? "+" : "-"} £{" "}
                            {Math.abs(amount).toFixed(2)}
                          </span>
                        </Table.Cell>
                        <Table.Cell className="py-4 px-6 text-sm text-[#8B8E98] whitespace-nowrap w-[25%] sm:w-[15%]">
                          {formatShortDate(tx.timestamp || tx.date || "")}
                        </Table.Cell>
                        <Table.Cell className="py-4 px-6 w-[25%] sm:w-[20%]">
                          <Chip
                            className={cn(
                              "px-3 py-1.5 rounded-md text-[10px] font-bold uppercase tracking-wider border",
                              getCategoryTheme(cat),
                            )}
                          >
                            {cat}
                          </Chip>
                        </Table.Cell>
                        <Table.Cell className="py-4 px-6 hidden sm:table-cell sm:w-[10%] text-center">
                          <div className="flex justify-center items-center w-full">
                            <Button
                              variant="ghost"
                              isIconOnly
                              onPress={() => {
                                setSelectedTx(tx);
                                setEditCategory(cat);
                              }}
                              className="text-[#5E6272] hover:text-neon-cyan hover:bg-neon-cyan/10 min-w-8 w-8 h-8 rounded-full cursor-pointer transition-colors flex items-center justify-center p-0 m-0"
                            >
                              <Pencil size={14} />
                            </Button>
                          </div>
                        </Table.Cell>
                      </Table.Row>
                    );
                  })}
                </Table.Body>
              </Table.Content>
            </Table.ScrollContainer>
          </Table>
        )}
      </Card.Content>

      <Modal
        isOpen={!!selectedTx}
        onOpenChange={(isOpen) => {
          if (!isOpen) setSelectedTx(null);
        }}
      >
        <Modal.Backdrop className="fixed inset-0 z-100 bg-black/60 backdrop-blur-sm">
          <Modal.Container className="fixed inset-0 z-101 flex items-center justify-center p-4">
            <Modal.Dialog className="bg-obsidian border border-white/8 rounded-2xl shadow-[0_0_50px_rgba(0,0,0,0.5)] p-6 relative max-w-md w-full pointer-events-auto backdrop-blur-xl">
              <Modal.CloseTrigger className="absolute top-4 right-4 text-[#8B8E98] hover:text-white transition-colors cursor-pointer" />
              <Modal.Header className="mb-6">
                <Modal.Heading className="text-xl font-bold text-white tracking-tight">
                  Transaction Details
                </Modal.Heading>
              </Modal.Header>
              <Modal.Body className="space-y-6">
                {selectedTx && (
                  <>
                    <div className="flex flex-col gap-1">
                      <Label className="text-[10px] font-bold text-[#5E6272] uppercase tracking-widest">
                        Description
                      </Label>
                      <p className="text-white text-base font-medium">
                        {selectedTx.description ||
                          selectedTx.Description ||
                          "Unknown"}
                      </p>
                    </div>
                    <div className="flex justify-between items-center bg-obsidian/40 p-4 rounded-2xl border border-white/8">
                      <div className="flex flex-col gap-1">
                        <Label className="text-[10px] font-bold text-[#5E6272] uppercase tracking-widest">
                          Amount
                        </Label>
                        <p
                          className={cn(
                            "text-2xl font-bold tracking-tight",
                            (selectedTx.amount ?? selectedTx.Amount ?? 0) > 0 ||
                              selectedTx.category === "Income" ||
                              selectedTx.Category === "Income"
                              ? "text-neon-cyan"
                              : "text-deep-pink",
                          )}
                        >
                          {(selectedTx.amount ?? selectedTx.Amount ?? 0) > 0 ||
                          selectedTx.category === "Income" ||
                          selectedTx.Category === "Income"
                            ? "+"
                            : "-"}{" "}
                          £
                          {Math.abs(
                            selectedTx.amount ?? selectedTx.Amount ?? 0,
                          ).toFixed(2)}
                        </p>
                      </div>
                      <div className="flex flex-col gap-1 text-right">
                        <Label className="text-[10px] font-bold text-[#5E6272] uppercase tracking-widest">
                          Date
                        </Label>
                        <p className="text-white font-medium text-sm">
                          {formatShortDate(
                            selectedTx.timestamp || selectedTx.date || "",
                          )}
                        </p>
                      </div>
                    </div>

                    <div className="flex flex-col gap-2 pt-2">
                      <Label className="text-sm font-bold text-white">
                        Edit Category
                      </Label>
                      <Dropdown>
                        <Dropdown.Trigger>
                          <div
                            role="button"
                            tabIndex={0}
                            className="w-full relative flex items-center justify-center bg-transparent hover:bg-white/5 border border-neon-cyan/70 text-white rounded-xl h-12 px-4 cursor-pointer transition-colors outline-none"
                          >
                            <ChevronDown
                              size={16}
                              className="absolute left-4 text-[#8B8E98] pointer-events-none"
                            />
                            <span className="font-medium pointer-events-none">
                              {editCategory || "Select Category"}
                            </span>
                          </div>
                        </Dropdown.Trigger>
                        <Dropdown.Popover
                          className="bg-obsidian border border-neon-cyan/70 shadow-2xl rounded-2xl w-50 min-w-50 backdrop-blur-xl"
                          placement="bottom left"
                        >
                          <Dropdown.Menu
                            items={STANDARD_CATEGORIES.map((c) => ({
                              id: c,
                              name: c,
                            }))}
                            selectionMode="single"
                            selectedKeys={new Set([editCategory])}
                            onSelectionChange={(keys: Selection) => {
                              if (keys !== "all") {
                                const selectedCat = Array.from(keys)[0];
                                if (selectedCat)
                                  setEditCategory(String(selectedCat));
                              }
                            }}
                            className="p-2 max-h-55 overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]"
                          >
                            {(cat: { id: string; name: string }) => (
                              <Dropdown.Item
                                key={cat.id}
                                textValue={cat.name}
                                className="rounded-xl transition-all data-[hover=true]:bg-white/10 py-3 px-4 outline-none cursor-pointer focus:ring-0 focus:outline-none w-full block"
                              >
                                <Badge.Anchor className="w-full relative flex items-center justify-between">
                                  <Label className="text-sm font-medium text-white cursor-pointer block w-full text-left pointer-events-none pr-4">
                                    {cat.name}
                                  </Label>
                                  {editCategory === cat.name && (
                                    <Badge className="bg-brand-green border-none w-2.5 h-2.5 min-w-0 p-0 relative transform-none rounded-full shrink-0" />
                                  )}
                                </Badge.Anchor>
                              </Dropdown.Item>
                            )}
                          </Dropdown.Menu>
                        </Dropdown.Popover>
                      </Dropdown>
                    </div>
                  </>
                )}
              </Modal.Body>
              <Modal.Footer className="flex justify-center gap-4 mt-6 pb-2">
                <Button
                  variant="ghost"
                  onPress={() => setSelectedTx(null)}
                  className="text-[#8B8E98] hover:text-white font-medium px-6 h-11 cursor-pointer bg-transparent border-none"
                >
                  Cancel
                </Button>
                <Button
                  variant="primary"
                  onPress={handleUpdateCategory}
                  isDisabled={isUpdating}
                  className="bg-neon-cyan hover:bg-neon-cyan/80 text-obsidian font-bold px-8 h-11 rounded-xl transition-colors cursor-pointer border-none shadow-[0_0_15px_rgba(0,229,255,0.4)]"
                >
                  {isUpdating ? "Retraining..." : "Update & Retrain"}
                </Button>
              </Modal.Footer>
            </Modal.Dialog>
          </Modal.Container>
        </Modal.Backdrop>
      </Modal>
    </Card>
  );
}
