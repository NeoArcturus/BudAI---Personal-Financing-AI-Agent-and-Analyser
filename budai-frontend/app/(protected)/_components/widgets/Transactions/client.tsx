"use client";

import React, { useState, useMemo } from "react";
import { ChevronDown, ListVideo, Pencil, Calendar as CalendarIcon } from "lucide-react";
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
  Spinner,
  ProgressBar,
  Switch,
} from "@heroui/react";
import { useBudAI } from "@/app/context/AppContext";
import { apiFetch } from "@/lib/api";
import { cn } from "@/lib/utils";
import { parseDate, CalendarDate } from "@internationalized/date";
import { useTransactions, usePersistedState, usePersistedDate } from "@/lib/hooks";
import WidgetFlipCard from "../../internal/FlipCard";
import { useRouter } from "next/navigation";
import { useQueryClient } from "@tanstack/react-query";
import { WidgetContext } from "../../../home/DashboardClient";

interface LedgerTableWidgetProps {
  initialData?: Transaction[];
}

type ExtendedTx = Transaction;

const STANDARD_CATEGORIES = [
  "Income",
  "Housing",
  "Food & Dining",
  "Transportation",
  "Utilities",
  "Entertainment & Lifestyle",
  "Subscriptions & Digital Services",
  "Shopping & Retail",
  "Healthcare",
  "Transfers & Payments",
  "Fees & Charges",
  "Savings & Investments",
  "Taxes & Government Payments",
  "Uncategorized"
];

export default function LedgerTableWidgetClient({
  initialData,
}: LedgerTableWidgetProps) {
  const router = useRouter();
  const { onRemove, instanceId } = React.useContext(WidgetContext);
  const queryClient = useQueryClient();
  const { accounts, createNewSession } = useBudAI();

  const [selectedAccountId, setSelectedAccountId] = usePersistedState<string>(
    `ledger_account${instanceId ? `-${instanceId}` : ""}`,
    accounts[0]?.account_id || "",
  );

  React.useEffect(() => {
    if (accounts.length > 0) {
      if (!selectedAccountId || selectedAccountId.startsWith("react-aria-") || !accounts.find(a => a.account_id === selectedAccountId)) {
        setSelectedAccountId(accounts[0].account_id);
      }
    }
  }, [accounts, selectedAccountId, setSelectedAccountId]);

  const [fromDate, setFromDate] = usePersistedDate(`tx_start_${instanceId || ""}`,
    parseDate(
      new Date(new Date().setDate(new Date().getDate() - 180))
        .toISOString()
        .split("T")[0],
    ),
  );
  const [toDate, setToDate] = usePersistedDate(`tx_end_${instanceId || ""}`,
    parseDate(new Date().toISOString().split("T")[0]),
  );

  const [selectedTx, setSelectedTx] = useState<ExtendedTx | null>(null);
  const [editCategory, setEditCategory] = useState<string>("");
  const [isUpdating, setIsUpdating] = useState<boolean>(false);
  const [isSurvivalMode, setIsSurvivalMode] = usePersistedState<boolean>(`tx_survival_${instanceId || ""}`, false);

  const { data: transactions = [], isLoading, isFetching } = useTransactions(
    selectedAccountId,
    fromDate?.toString(),
    toDate?.toString(),
    initialData,
  );

  const filteredTransactions = React.useMemo(() => {
    let list = Array.isArray(transactions) ? transactions : [];
    if (isSurvivalMode) {
      list = list.filter((tx) => {
        const tags = tx.tags || [];
        return tags.includes("#essential") || tags.includes("#housing") || tags.includes("#recurring");
      });
    }
    return list;
  }, [transactions, isSurvivalMode]);

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
      data: Array.isArray(transactions) ? transactions.slice(0, 15) : [],
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
      <Card className="w-full h-full  font-geist">
        <Card.Header className="flex flex-col p-6 border-b border-white/5 shrink-0 gap-4 w-full z-10">
          <div className="flex justify-between items-center w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
              Recent Transactions
            </h3>
            <div className="flex items-center gap-4">
              <Button size="sm" variant="ghost" className="text-[10px] font-black text-foreground/50 hover:text-foreground uppercase tracking-widest bg-transparent border-none">
                View All
              </Button>
              <CloseButton
                onPress={onRemove}
                className="w-8 h-8 min-w-8 opacity-50 hover:opacity-100 hover:bg-white/10 text-foreground transition-all rounded-full"
              />
            </div>
          </div>
        </Card.Header>

        <Card.Content className="flex-1 h-full w-full p-0 overflow-hidden relative">
          {(isLoading || !Array.isArray(transactions) || transactions.length === 0) ? (
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
          ) : (
            <div className="relative w-full h-full">
              {isFetching && (
                <div className="absolute top-0 left-0 w-full z-30 pointer-events-none flex items-center">
                  <ProgressBar isIndeterminate aria-label="Loading..." size="sm" color="accent" className="w-full" />
                  <div className="absolute top-2 right-6">
                    <Spinner size="sm" color="accent" />
                  </div>
                </div>
              )}
              <div className="flex flex-col w-full h-full overflow-y-auto custom-scrollbar px-6 pb-6 pt-2">
                {filteredTransactions.map((tx, i) => {
                  const displayDesc = tx.merchant_name || tx.description || "Unknown";
                  const amount = tx.amount ?? 0;
                  const cat = tx.category || "Uncategorized";
                  const initial = displayDesc.charAt(0).toUpperCase();
                  const isPositive = amount > 0 || cat === "Income";

                  const formattedAmount = new Intl.NumberFormat("en-US", {
                    style: "currency",
                    currency: activeAccount?.currency || "USD",
                    minimumFractionDigits: 2,
                    maximumFractionDigits: 2,
                  }).format(Math.abs(amount));

                  // Using standard colors to match image
                  let iconColor = isPositive ? "text-green-500 bg-green-500/10" : "text-orange-500 bg-orange-500/10";

                  // We'll mimic the brands based on initial for a closer match to the image
                  if (displayDesc.toLowerCase().includes("netflix")) iconColor = "text-red-500 bg-red-500/10";
                  if (displayDesc.toLowerCase().includes("spotify")) iconColor = "text-green-500 bg-green-500/10";
                  if (displayDesc.toLowerCase().includes("amazon")) iconColor = "text-orange-500 bg-orange-500/10";

                  return (
                    <div
                      key={i}
                      onClick={() => {
                        setSelectedTx(tx);
                        setEditCategory(cat);
                      }}
                      className="flex justify-between items-center py-4 border-b border-white/5 last:border-0 hover:bg-white/5 cursor-pointer transition-colors w-full group"
                    >
                      <div className="flex items-center gap-4">
                        <div className={cn("w-10 h-10 rounded-xl flex items-center justify-center text-sm font-bold shadow-sm", iconColor)}>
                          {initial}
                        </div>
                        <div className="flex flex-col gap-1 text-left">
                          <span className="font-bold text-sm tracking-wide text-foreground group-hover:text-primary transition-colors">
                            {displayDesc}
                          </span>
                          <span className="text-[10px] font-mono text-foreground/50 uppercase tracking-widest">
                            {formatShortDate(tx.timestamp || tx.date || "")}
                          </span>
                        </div>
                      </div>
                      <div className="flex flex-col items-end gap-1">
                        <span
                          className={cn(
                            "font-mono text-sm tracking-tighter font-bold",
                            isPositive ? "text-green-500" : "text-red-500",
                          )}
                        >
                          {isPositive ? "+" : "-"}{formattedAmount}
                        </span>
                        <span className="text-[9px] font-mono uppercase tracking-widest font-black text-green-500/80">
                          Completed
                        </span>
                      </div>
                    </div>
                  );
                })}
              </div>
            </div>
          )}
        </Card.Content>

        <Modal.Backdrop
          isOpen={!!selectedTx}
          onOpenChange={(isOpen) => {
            if (!isOpen) setSelectedTx(null);
          }}
          variant="blur"
        >
            <Modal.Container placement="center" >
              <Modal.Dialog className="modal p-8 relative max-w-md w-full pointer-events-auto bg-black/90 border border-white/5 shadow-2xl rounded-2xl flex flex-col">
                <Modal.CloseTrigger className="absolute top-6 right-6 text-foreground/40 hover:text-foreground transition-colors cursor-pointer" />
                <Modal.Header className="mb-6 flex justify-center">
                  <Modal.Heading className="text-[10px] font-black uppercase tracking-[0.4em] italic text-primary text-center">
                    Transactions Details
                  </Modal.Heading>
                </Modal.Header>
                <Modal.Body className="space-y-0 p-0">
                  {selectedTx && (
                    <>
                      <div className="flex flex-col items-center justify-center border-b border-white/5 pb-8 mb-6">
                        <p
                          className={cn(
                            "text-4xl font-mono tracking-tighter font-bold",
                            (selectedTx.amount ?? 0) > 0 || selectedTx.category === "Income"
                              ? "text-white"
                              : "text-foreground/80",
                          )}
                        >
                          {(selectedTx.amount ?? 0) > 0 || selectedTx.category === "Income" ? "+" : "-"}{" "}
                          {new Intl.NumberFormat("en-GB", {
                            style: "currency",
                            currency: activeAccount?.currency || "GBP",
                            minimumFractionDigits: 2,
                            maximumFractionDigits: 2,
                          }).format(Math.abs(selectedTx.amount ?? 0))}
                        </p>
                      </div>

                      <div className="grid grid-cols-2 gap-y-6 gap-x-4 mb-8">
                        <div className="flex flex-col gap-1">
                          <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 block">
                            Date
                          </Label>
                          <p className="font-mono text-sm tracking-tight text-foreground font-medium">
                            {formatShortDate(selectedTx.timestamp || selectedTx.date || "")}
                          </p>
                        </div>
                        <div className="flex flex-col gap-1 text-right">
                          <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 block">
                            Status
                          </Label>
                          <p className="font-mono text-sm tracking-tight text-foreground font-medium text-green-500/80">
                            CLEARED
                          </p>
                        </div>
                        <div className="flex flex-col gap-1 col-span-2">
                          <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 block">
                            Merchant
                          </Label>
                          <p className="font-mono text-sm tracking-tight text-foreground font-medium">
                            {selectedTx.merchant_name || selectedTx.description || "Unknown"}
                          </p>
                          {selectedTx.merchant_name && selectedTx.description && selectedTx.merchant_name !== selectedTx.description && (
                            <p className="text-[10px] font-mono text-foreground/20 mt-1 uppercase tracking-wider">
                              RAW: {selectedTx.description}
                            </p>
                          )}
                        </div>
                      </div>

                      <div className="flex flex-col gap-2 pt-2 border-t border-white/5 mt-2">
                        <Label className="text-[9px] font-black uppercase tracking-[0.3em] text-foreground/40 mt-4 block">
                          Category
                        </Label>
                        <Dropdown>
                          <Dropdown.Trigger className="w-56 relative flex items-center justify-between bg-white/5 hover:bg-white/10 border border-white/10 text-foreground rounded-lg h-12 px-4 cursor-pointer transition-colors outline-none">
                            <span className="font-mono text-xs font-bold tracking-tight pointer-events-none">
                              {editCategory || "Select Category"}
                            </span>
                            <ChevronDown
                              size={14}
                              className="text-foreground/40 pointer-events-none"
                            />
                          </Dropdown.Trigger>
                          <Dropdown.Popover
                            className="bg-black/95 backdrop-blur-3xl border border-white/10 shadow-2xl rounded-xl w-56"
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
                                  id={cat.id}
                                  textValue={cat.name}
                                  className="rounded-lg transition-all data-[hover=true]:bg-white/10 py-3 px-4 outline-none cursor-pointer focus:ring-0 focus:outline-none w-full block"
                                >
                                  <div className="w-full relative flex items-center justify-between pointer-events-none">
                                    <span className="text-xs font-mono font-bold text-foreground">
                                      {cat.name}
                                    </span>
                                    {editCategory === cat.name && (
                                      <div className="bg-primary w-2 h-2 rounded-full shadow-[0_0_8px_rgba(0,242,255,0.8)]" />
                                    )}
                                  </div>
                                </Dropdown.Item>
                              )}
                            </Dropdown.Menu>
                          </Dropdown.Popover>
                        </Dropdown>
                      </div>

                      <div className="mt-8 pt-4 border-t border-white/5 flex flex-col gap-4">
                        <Button
                          variant="primary"
                          onPress={handleUpdateCategory}
                          isDisabled={isUpdating}
                          className="w-full bg-primary/10 hover:bg-primary/20 text-primary border border-primary/30 font-mono text-xs font-bold tracking-widest px-8 h-12 rounded-lg transition-all cursor-pointer shadow-[0_0_15px_rgba(0,242,255,0.1)]"
                        >
                          {isUpdating ? "PROCESSING..." : "CHANGE CATEGORY"}
                        </Button>
                        <div className="flex justify-center w-full">
                          <span className="text-[8px] font-mono tracking-widest text-foreground/20 uppercase">
                            TXN-UUID: {selectedTx.transaction_id || "N/A"}
                          </span>
                        </div>
                      </div>
                    </>
                  )}
                </Modal.Body>
              </Modal.Dialog>
            </Modal.Container>
          </Modal.Backdrop>
      </Card>
    </WidgetFlipCard>
  );
}
