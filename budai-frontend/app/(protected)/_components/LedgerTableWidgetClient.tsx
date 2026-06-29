"use client";

import React, { useState, useMemo } from "react";
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
  DatePicker,
  DateField,
  Calendar,
  Modal,
  Badge,
  Skeleton,
  CloseButton,
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";
import { parseDate, CalendarDate } from "@internationalized/date";
import { useTransactions, usePersistedState } from "@/lib/hooks";
import WidgetFlipCard from "./WidgetFlipCard";
import { useRouter } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { WidgetContext } from "../home/DashboardClient";

interface LedgerTableWidgetProps {
  initialData?: Transaction[];
}

type ExtendedTx = Transaction;

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

export default function LedgerTableWidgetClient({
  initialData,
}: LedgerTableWidgetProps) {
  const router = useRouter();
  const { onRemove } = React.useContext(WidgetContext);
  const queryClient = useQueryClient();
  const { accounts, createNewSession } = useBudAI();

  const [selectedAccountId, setSelectedAccountId] = usePersistedState<string>(
    "ledger_account",
    accounts[0]?.account_id || "",
  );

  const [fromDate, setFromDate] = useState<CalendarDate | null>(
    parseDate(
      new Date(new Date().setDate(new Date().getDate() - 180))
        .toISOString()
        .split("T")[0],
    ),
  );
  const [toDate, setToDate] = useState<CalendarDate | null>(
    parseDate(new Date().toISOString().split("T")[0]),
  );

  const [selectedTx, setSelectedTx] = useState<ExtendedTx | null>(null);
  const [editCategory, setEditCategory] = useState<string>("");
  const [isUpdating, setIsUpdating] = useState<boolean>(false);

  const { data: transactions = [], isLoading } = useTransactions(
    selectedAccountId,
    fromDate?.toString(),
    toDate?.toString(),
    initialData,
  );

  const handleUpdateCategory = async () => {
    if (!selectedTx || !editCategory) return;

    const txId = selectedTx.transaction_uuid || selectedTx.transaction_id;

    if (!txId) return;

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
        true,
      );

      if (res.ok) {
        const data = await res.json() as any;
        const taskId = data.task_id;

        if (taskId) {
          const pollInterval = setInterval(async () => {
            try {
              const statusRes = await apiFetch(
                `/api/categorizer/task-status/${taskId}`,
                {},
                true,
              );
              if (statusRes.ok) {
                const statusData = await statusRes.json() as any;
                if (
                  statusData.status === "completed" ||
                  statusData.status === "failed"
                ) {
                  clearInterval(pollInterval);
                  queryClient.invalidateQueries({ queryKey: ["transactions"] });
                  setIsUpdating(false);
                }
              }
            } catch (e) {
              console.log(e);
              clearInterval(pollInterval);
              setIsUpdating(false);
            }
          }, 2000);
        } else {
          queryClient.invalidateQueries({ queryKey: ["transactions"] });
          setIsUpdating(false);
        }

        setSelectedTx(null);
      }
    } catch (e) {
      console.error(e);
      setIsUpdating(false);
    }
  };

  const handleDiscuss = () => {
    const sessionId = createNewSession("Transaction Audit Session", {
      type: "ledger_audit",
      accountId: selectedAccountId,
      data: transactions.slice(0, 15),
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

  const getCategoryTheme = (category: string) => {
    const themes: Record<string, string> = {
      "Food & Dining": "bg-pink-500/20 text-pink-500 border-pink-500/40",
      Groceries: "bg-orange-500/20 text-orange-500 border-orange-500/40",
      Transportation: "bg-yellow-500/20 text-yellow-500 border-yellow-500/40",
      "Bills & Utilities": "bg-cyan-300/20 text-cyan-400 border-cyan-300/40",
      Rent: "bg-blue-500/20 text-blue-500 border-blue-500/40",
      Shopping: "bg-purple-500/20 text-purple-500 border-purple-500/40",
      Entertainment: "bg-destructive/20 text-destructive border-destructive/40",
      "Health & Wellness": "bg-green-500/20 text-green-500 border-green-500/40",
      "Transfers & Investments":
        "bg-emerald-500/20 text-emerald-500 border-emerald-500/40",
      "High-Risk / Anomaly": "bg-red-500/20 text-red-500 border-red-500/40",
      Salary: "bg-green-500/20 text-green-500 border-green-500/40",
      Income: "bg-green-500/20 text-green-500 border-green-500/40",
      Travel: "bg-yellow-500/20 text-yellow-500 border-yellow-500/40",
      Education: "bg-lime-500/20 text-lime-500 border-lime-500/40",
    };
    return (
      themes[category] ||
      "bg-muted text-muted-foreground border-muted-foreground/40"
    );
  };

  const formatShortDate = (dateStr: string) => {
    if (!dateStr || dateStr === "NaT") return "Processing...";
    const d = new Date(dateStr);
    if (isNaN(d.getTime())) return "Processing...";
    const day = d.toLocaleDateString("en-US", { weekday: "short" });
    const time = d
      .toLocaleTimeString("en-US", { hour: "numeric", minute: "2-digit" })
      .toLowerCase();
    return `${day} ${time}`;
  };

  return (
    <WidgetFlipCard
      insight={undefined}
      isLoading={false}
      isDataLoading={isLoading}
      onDiscuss={handleDiscuss}
    >
      <Card className="w-full h-full liquid-glass rounded-3xl flex flex-col overflow-hidden relative font-geist">
        <Card.Header className="flex flex-col p-8 border-b-[0.5px] border-white/5 shrink-0 gap-6 w-full z-10">
          <div className="flex justify-between items-center w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
              Transaction history
            </h3>
            <CloseButton
              onPress={onRemove}
              className="text-foreground/20 hover:text-foreground transition-all rounded-md"
            />
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
                className="bg-white/5 border border-white/10 rounded-xl px-3 h-10 flex items-center transition-colors focus-within:border-primary/50"
              >
                <DateField.Input className="flex-1 bg-transparent text-foreground text-sm outline-none">
                  {(segment) => (
                    <DateField.Segment
                      segment={segment}
                      className="focus:bg-primary/20 rounded px-0.5 outline-none"
                    />
                  )}
                </DateField.Input>
                <DateField.Suffix className="ml-2 flex items-center">
                  <DatePicker.Trigger className="text-muted-foreground hover:text-foreground cursor-pointer transition-colors">
                    <DatePicker.TriggerIndicator />
                  </DatePicker.Trigger>
                </DateField.Suffix>
              </DateField.Group>
              <DatePicker.Popover className="bg-black/60 backdrop-blur-3xl border border-white/10 rounded-2xl p-4 shadow-2xl z-50">
                <Calendar aria-label="From date" className="w-full min-w-65">
                  <Calendar.Header className="flex items-center gap-2 mb-4">
                    <Calendar.YearPickerTrigger className="flex items-center gap-1 mr-auto cursor-pointer hover:opacity-80 transition-opacity">
                      <Calendar.YearPickerTriggerHeading className="text-base font-semibold text-primary" />
                      <Calendar.YearPickerTriggerIndicator className="text-muted-foreground w-4 h-4" />
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
                        <Calendar.HeaderCell className="text-xs font-medium text-muted-foreground pb-3 text-center">
                          {day}
                        </Calendar.HeaderCell>
                      )}
                    </Calendar.GridHeader>
                    <Calendar.GridBody>
                      {(date) => (
                        <Calendar.Cell
                          date={date}
                          className="w-8 h-8 flex items-center justify-center mx-auto text-sm text-foreground rounded-full hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-colors"
                        />
                      )}
                    </Calendar.GridBody>
                  </Calendar.Grid>
                </Calendar>
              </DatePicker.Popover>
            </DatePicker>

            <span className="text-muted-foreground">-</span>

            <DatePicker
              className="w-50 sm:min-w-35 sm:max-w-40"
              name="To Date"
              value={toDate}
              onChange={setToDate}
            >
              <DateField.Group
                fullWidth
                className="bg-white/5 border border-white/10 rounded-xl px-3 h-10 flex items-center transition-colors focus-within:border-primary/50"
              >
                <DateField.Input className="flex-1 bg-transparent text-foreground text-sm outline-none">
                  {(segment) => (
                    <DateField.Segment
                      segment={segment}
                      className="focus:bg-primary/20 rounded px-0.5 outline-none"
                    />
                  )}
                </DateField.Input>
                <DateField.Suffix className="ml-2 flex items-center">
                  <DatePicker.Trigger className="text-muted-foreground hover:text-foreground cursor-pointer transition-colors">
                    <DatePicker.TriggerIndicator />
                  </DatePicker.Trigger>
                </DateField.Suffix>
              </DateField.Group>
              <DatePicker.Popover className="bg-black/60 backdrop-blur-3xl border border-white/10 rounded-2xl p-4 shadow-2xl z-50">
                <Calendar aria-label="To date" className="w-full min-w-65">
                  <Calendar.Header className="flex items-center gap-2 mb-4">
                    <Calendar.YearPickerTrigger className="flex items-center gap-1 mr-auto cursor-pointer hover:opacity-80 transition-opacity">
                      <Calendar.YearPickerTriggerHeading className="text-base font-semibold text-primary" />
                      <Calendar.YearPickerTriggerIndicator className="text-muted-foreground w-4 h-4" />
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
                        <Calendar.HeaderCell className="text-xs font-medium text-muted-foreground pb-3 text-center">
                          {day}
                        </Calendar.HeaderCell>
                      )}
                    </Calendar.GridHeader>
                    <Calendar.GridBody>
                      {(date) => (
                        <Calendar.Cell
                          date={date}
                          className="w-8 h-8 flex items-center justify-center mx-auto text-sm text-foreground rounded-full hover:bg-white/10 data-[selected=true]:bg-primary data-[selected=true]:text-primary-foreground cursor-pointer outline-none transition-colors"
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
        </Card.Header>

        <Card.Content className="flex-1 h-full w-full p-0 overflow-hidden">
          {isLoading ? (
            <div className="w-full h-full px-6 py-4 space-y-4 overflow-hidden">
              <div className="flex items-center justify-between border-b border-border pb-4">
                <Skeleton
                  animationType="shimmer"
                  className="h-3 w-1/4 rounded bg-secondary"
                />
                <Skeleton
                  animationType="shimmer"
                  className="h-3 w-1/6 rounded bg-secondary"
                />
                <Skeleton
                  animationType="shimmer"
                  className="h-3 w-1/6 rounded bg-secondary"
                />
                <Skeleton
                  animationType="shimmer"
                  className="h-3 w-1/6 rounded bg-secondary"
                />
              </div>
              {Array.from({ length: 8 }).map((_, i) => (
                <div
                  key={i}
                  className="flex items-center justify-between gap-4 py-2"
                >
                  <div className="flex items-center gap-4 w-1/4 sm:w-[40%]">
                    <Skeleton
                      animationType="shimmer"
                      className="w-9 h-9 rounded-full shrink-0 bg-secondary"
                    />
                    <Skeleton
                      animationType="shimmer"
                      className="h-4 w-full rounded bg-secondary hidden sm:block"
                    />
                  </div>
                  <Skeleton
                    animationType="shimmer"
                    className="h-4 w-[30%] sm:w-[15%] rounded bg-secondary"
                  />
                  <Skeleton
                    animationType="shimmer"
                    className="h-4 w-[25%] sm:w-[15%] rounded bg-secondary"
                  />
                  <Skeleton
                    animationType="shimmer"
                    className="h-6 w-[25%] sm:w-[20%] rounded-md bg-secondary"
                  />
                </div>
              ))}
            </div>
          ) : transactions.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full w-full opacity-50">
              <div className="w-12 h-12 rounded-full border-2 border-dashed border-muted-foreground flex items-center justify-center mb-3">
                <ListVideo size={20} className="text-muted-foreground" />
              </div>
              <span className="text-muted-foreground text-sm font-medium">
                No transactions found.
              </span>
            </div>
          ) : (
            <Table className="w-full h-full text-left relative table-fixed">
              <Table.ScrollContainer className="h-full w-full overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] px-2 pb-4">
                <Table.Content
                  aria-label="Transaction Ledger"
                  className="w-full"
                >
                  <Table.Header className="sticky top-0 bg-black/40 backdrop-blur-xl z-20 border-b-[0.5px] border-white/5 w-full">
                    <Table.Column
                      isRowHeader
                      className="py-5 px-8 text-[9px] font-black uppercase tracking-[0.3em] text-foreground/30 w-[20%] sm:w-[40%]"
                    >
                      Description
                    </Table.Column>
                    <Table.Column className="py-5 px-8 text-[9px] font-black uppercase tracking-[0.3em] text-foreground/30 w-[30%] sm:w-[15%]">
                      Amount
                    </Table.Column>
                    <Table.Column className="py-5 px-8 text-[9px] font-black uppercase tracking-[0.3em] text-foreground/30 w-[25%] sm:w-[15%]">
                      Date
                    </Table.Column>
                    <Table.Column className="py-5 px-8 text-[9px] font-black uppercase tracking-[0.3em] text-foreground/30 w-[25%] sm:w-[20%]">
                      Category
                    </Table.Column>
                    <Table.Column className="py-5 px-8 text-[9px] font-black uppercase tracking-[0.3em] text-foreground/30 hidden sm:table-cell sm:w-[10%] text-center">
                      Operation
                    </Table.Column>
                  </Table.Header>
                  <Table.Body className="w-full divide-y-[0.5px] divide-white/5">
                    {transactions.map((tx, i) => {
                      const desc = tx.description || "UNDEFINED_ENTITY";
                      const amount = tx.amount ?? 0;
                      const cat = tx.category || "UNCATEGORIZED";
                      const initial = desc.charAt(0).toUpperCase();
                      const isPositive = amount > 0 || cat === "Income";

                      return (
                        <Table.Row
                          key={i}
                          className="hover:bg-white/3 transition-all group border-b-[0.5px] border-white/5 last:border-0 w-full cursor-pointer sm:cursor-default"
                          onClick={() => {
                            if (window.innerWidth < 640) {
                              setSelectedTx(tx);
                              setEditCategory(cat);
                            }
                          }}
                        >
                          <Table.Cell className="py-5 px-8 w-[20%] sm:w-[40%]">
                            <div className="flex items-center gap-5 w-full">
                              <div
                                className={cn(
                                  "w-8 h-8 shrink-0 rounded-lg flex items-center justify-center border-[0.5px] text-[11px] font-black shadow-sm",
                                  isPositive
                                    ? "border-green-500/40 bg-green-500/10 text-green-500"
                                    : "border-primary/40 bg-primary/10 text-primary",
                                )}
                              >
                                {initial}
                              </div>
                              <span className="hidden sm:block text-foreground text-[11px] font-black uppercase tracking-tight truncate w-full group-hover:text-primary transition-colors italic">
                                {desc}
                              </span>
                            </div>
                          </Table.Cell>
                          <Table.Cell className="py-5 px-8 w-[30%] sm:w-[15%]">
                            <span
                              className={cn(
                                "text-[12px] font-black font-mono tracking-tighter",
                                isPositive
                                  ? "text-green-500/60"
                                  : "text-red-500/60",
                              )}
                            >
                              {isPositive ? "+" : "-"}£
                              {Math.abs(amount).toFixed(2)}
                            </span>
                          </Table.Cell>
                          <Table.Cell className="py-5 px-8 text-[10px] font-bold font-mono text-foreground/30 uppercase tracking-widest whitespace-nowrap w-[25%] sm:w-[15%]">
                            {new Date(
                              tx.timestamp || tx.date || "",
                            ).toLocaleDateString("en-GB", {
                              day: "2-digit",
                              month: "2-digit",
                              year: "2-digit",
                            })}
                          </Table.Cell>
                          <Table.Cell className="py-5 px-8 w-[25%] sm:w-[20%]">
                            <div
                              className={cn(
                                "px-3 py-1 rounded-lg text-[9px] font-black uppercase tracking-widest border-[0.5px] w-fit shadow-sm",
                                getCategoryTheme(cat),
                              )}
                            >
                              {cat}
                            </div>
                          </Table.Cell>
                          <Table.Cell className="py-5 px-8 hidden sm:table-cell sm:w-[10%] text-center">
                            <Button
                              variant="ghost"
                              isIconOnly
                              onPress={() => {
                                setSelectedTx(tx);
                                setEditCategory(cat);
                              }}
                              className="text-foreground/20 hover:text-primary hover:bg-white/5 w-8 h-8 rounded-lg cursor-pointer transition-all opacity-0 group-hover:opacity-100 p-0 flex items-center justify-center"
                            >
                              <Pencil size={14} />
                            </Button>
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
          <Modal.Backdrop className="fixed inset-0 z-100 bg-black/40 backdrop-blur-md">
            <Modal.Container className="fixed inset-0 z-101 flex items-center justify-center p-4">
              <Modal.Dialog className="liquid-glass border-cyan-400/60 border rounded-xl shadow-2xl p-6 relative max-w-md w-full pointer-events-auto">
                <Modal.CloseTrigger className="absolute top-4 right-4 text-muted-foreground hover:text-foreground transition-colors cursor-pointer" />
                <Modal.Header className="mb-6">
                  <Modal.Heading className="text-2xl font-bold text-foreground tracking-tight">
                    Transaction Details
                  </Modal.Heading>
                </Modal.Header>
                <Modal.Body className="space-y-6">
                  {selectedTx && (
                    <>
                      <div className="flex flex-col gap-1">
                        <Label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">
                          Description
                        </Label>
                        <p className="text-foreground text-base font-medium">
                          {selectedTx.description || "Unknown"}
                        </p>
                      </div>
                      <div className="flex justify-between items-center bg-secondary/50 p-4 rounded-lg border border-border">
                        <div className="flex flex-col gap-1">
                          <Label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">
                            Amount
                          </Label>
                          <p
                            className={cn(
                              "text-2xl font-bold tracking-tight",
                              (selectedTx.amount ?? 0) > 0 ||
                                selectedTx.category === "Income"
                                ? "text-green-500"
                                : "text-destructive",
                            )}
                          >
                            {(selectedTx.amount ?? 0) > 0 ||
                            selectedTx.category === "Income"
                              ? "+"
                              : "-"}{" "}
                            £{Math.abs(selectedTx.amount ?? 0).toFixed(2)}
                          </p>
                        </div>
                        <div className="flex flex-col gap-1 text-right">
                          <Label className="text-[10px] font-bold text-muted-foreground uppercase tracking-widest">
                            Date
                          </Label>
                          <p className="text-foreground font-medium text-sm">
                            {formatShortDate(
                              selectedTx.timestamp || selectedTx.date || "",
                            )}
                          </p>
                        </div>
                      </div>

                      <div className="flex flex-col gap-2 pt-2">
                        <Label className="text-sm font-bold text-foreground">
                          Edit Category
                        </Label>
                        <Dropdown>
                          <Dropdown.Trigger className="w-50 relative flex items-center justify-center bg-transparent hover:bg-secondary/50 border border-primary/70 text-foreground rounded-md h-12 px-4 cursor-pointer transition-colors outline-none">
                            <ChevronDown
                              size={16}
                              className="absolute left-4 text-muted-foreground pointer-events-none"
                            />
                            <span className="font-medium pointer-events-none">
                              {editCategory || "Select Category"}
                            </span>
                          </Dropdown.Trigger>
                          <Dropdown.Popover
                            className="bg-popover border border-primary/70 shadow-2xl rounded-xl w-50 min-w-50"
                            placement="bottom"
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
                                  className="rounded-xl transition-all data-[hover=true]:bg-secondary py-3 px-4 outline-none cursor-pointer focus:ring-0 focus:outline-none w-full block"
                                >
                                  <Badge.Anchor className="w-full relative flex items-center justify-between">
                                    <Label className="text-sm font-medium text-foreground cursor-pointer block w-full text-left pointer-events-none pr-4">
                                      {cat.name}
                                    </Label>
                                    {editCategory === cat.name && (
                                      <Badge className="bg-green-500 border-none w-2.5 h-2.5 min-w-0 p-0 relative transform-none rounded-full shrink-0" />
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
                    className="text-muted-foreground hover:text-foreground font-medium px-6 h-11 cursor-pointer bg-transparent border-none"
                  >
                    Cancel
                  </Button>
                  <Button
                    variant="primary"
                    onPress={handleUpdateCategory}
                    isDisabled={isUpdating}
                    className="bg-primary hover:bg-primary/80 text-primary-foreground font-bold px-8 h-11 rounded-xl transition-colors cursor-pointer border-none shadow-[0_0_15px_rgba(0,127,255,0.4)]"
                  >
                    {isUpdating ? "Retraining..." : "Update & Retrain"}
                  </Button>
                </Modal.Footer>
              </Modal.Dialog>
            </Modal.Container>
          </Modal.Backdrop>
        </Modal>
      </Card>
    </WidgetFlipCard>
  );
}
