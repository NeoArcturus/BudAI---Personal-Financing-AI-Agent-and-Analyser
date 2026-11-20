"use client";

import React, {
  createContext,
  useContext,
  useState,
  useEffect,
  useCallback,
  useRef,
  useMemo,
} from "react";
import {
  Button,
  Modal,
  Select,
  ListBox,
  Label,
  Key,
  Selection,
  toast,
} from "@heroui/react";
import { useRouter } from "next/navigation";
import { UIMessage } from "@ai-sdk/react";
import {
  Account,
  Transaction,
  NativeChartConfig,
  TabType,
  ToolParameters,
  BankChartData,
  ParsedCategory,
  ExplanationState,
  ExplanationContextType,
  ExplanationPayload,
} from "@/types";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { apiFetch, getAuthToken, getApiUrl } from "@/lib/api";
import { useAccounts, useChatSessions, useDashboardSnapshot } from "@/lib/hooks";
import { useQueryClient } from "@tanstack/react-query";
import { Wallet } from "lucide-react";

interface ServerMessage {
  id?: string;
  role: "user" | "assistant" | "system" | "data";
  content: string;
  timestamp: string;
}

interface SessionResponse {
  messages: ServerMessage[];
}

export type AdvisorContextData =
  | Transaction[]
  | BankChartData[]
  | ParsedCategory[]
  | Record<string, string | number | boolean | null>
  | string
  | null;

export interface AdvisorContext {
  type:
    | "cash_flow"
    | "spending_trend"
    | "expense_distribution"
    | "ledger_audit"
    | "market_audit"
    | "general";
  accountId?: string;
  data?: AdvisorContextData;
}

export interface ChatSession {
  id: string;
  title: string;
  messages: UIMessage[];
  lastUpdated: Date;
  contextData?: AdvisorContext;
}

interface BudAIContextType {
  accounts: Account[];
  activeAccountId: string | null;
  setActiveAccountId: (id: string | null) => void;
  chartConfig: NativeChartConfig | null;
  setChartConfig: (config: NativeChartConfig | null) => void;
  isGenerating: boolean;
  setIsGenerating: (val: boolean) => void;
  handleAiChartTrigger: (
    type: TabType | string,
    customParams?: Partial<ToolParameters>,
  ) => Promise<void>;
  userName: string;
  totalBalance: number;
  explanation: ExplanationState;
  setExplanation: React.Dispatch<React.SetStateAction<ExplanationState>>;
  triggerExplanation: (
    type: ExplanationContextType,
    data: ExplanationPayload | null,
  ) => void;
  isChatOpen: boolean;
  setIsChatOpen: React.Dispatch<React.SetStateAction<boolean>>;
  currentSessionId: string | null;
  setCurrentSessionId: (id: string | null) => void;
  sessions: ChatSession[];
  createNewSession: (title: string, contextData?: AdvisorContext) => string;
  deleteSession: (id: string) => void;
  isChatLoading: boolean;
  chatMessages: UIMessage[];
  sendChatMessage: (
    textToSend: string,
    sessionId?: string,
  ) => Promise<string | undefined>;
  thinkingProfile: string | null;
  isAccountSelectorOpen: boolean;
  setIsAccountSelectorOpen: (val: boolean) => void;
  selectedAccountIds: Key[];
  setSelectedAccountIds: (ids: Key[]) => void;
  snapshot: any | null;
  snapshotDays: number;
  setSnapshotDays: (days: number) => void;
  snapshotPage: number;
  setSnapshotPage: (page: number) => void;
  isSnapshotFetching: boolean;
}

const BudAIContext = createContext<BudAIContextType | undefined>(undefined);

export const BudAIProvider = ({
  children,
  initialAccounts = [],
  initialSessions = [],
  initialSnapshot = null,
}: {
  children: React.ReactNode;
  initialAccounts?: Account[];
  initialSessions?: ChatSession[];
  initialSnapshot?: any;
}) => {
  const router = useRouter();
  const queryClient = useQueryClient();
  const [activeAccountId, setActiveAccountId] = useState<string | null>("ALL");
  const [chartConfig, setChartConfig] = useState<NativeChartConfig | null>(
    null,
  );
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const [isChatOpen, setIsChatOpen] = useState<boolean>(false);
  const [localSessions, setLocalSessions] =
    useState<ChatSession[]>(initialSessions);
  const [thinkingProfile, setThinkingProfile] = useState<string | null>(null);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);

  const [userName, setUserName] = useState<string>("Verified User");

  useEffect(() => {
    const stored = localStorage.getItem("budai_user_name");
    if (stored) setUserName(stored);

    const handleUnauthorized = () => {
      queryClient.clear();
      sessionStorage.clear();
      document.cookie = "budai_token=; path=/; max-age=0";
      localStorage.removeItem("budai_token");
      localStorage.removeItem("budai_user_name");

    };
    window.addEventListener("budai-unauthorized", handleUnauthorized);
    return () => window.removeEventListener("budai-unauthorized", handleUnauthorized);
  }, [router]);

  const [snapshotDays, setSnapshotDays] = useState<number>(30);
  const [snapshotPage, setSnapshotPage] = useState<number>(1);

  const handleSetSnapshotDays = (days: number) => {
    setSnapshotDays(days);
    setSnapshotPage(1);
  };

  const { data: snapshot = initialSnapshot, isFetching: isSnapshotFetching } = useDashboardSnapshot(
    snapshotDays,
    snapshotPage,
    50,
    initialSnapshot,
  );
  const accountsFromSnapshot: Account[] =
    snapshot?.financial_anchor?.connected_accounts ||
    initialSnapshot?.financial_anchor?.connected_accounts ||
    initialAccounts;
  const { data: fetchedAccounts = accountsFromSnapshot } = useAccounts(accountsFromSnapshot);
  const { data: serverSessions = [] } = useChatSessions();

  useEffect(() => {
    if (serverSessions.length > 0 && localSessions.length === 0) {
      setLocalSessions(
        serverSessions.map((s) => ({
          id: s.session_id,
          title: s.title,
          messages: [],
          lastUpdated: new Date(s.last_updated),
          contextData: s.context_data,
        })),
      );
    }
  }, [serverSessions, localSessions.length]);

  const totalBalance = useMemo(() => {
    return fetchedAccounts.reduce((sum: number, acc: Account) => sum + (acc.balance ?? 0), 0);
  }, [fetchedAccounts]);

  const activeAccount = fetchedAccounts.find(
    (a) => a.account_id === activeAccountId,
  );
  const bankName =
    activeAccountId === "ALL"
      ? "ALL"
      : activeAccount?.bank_name || "Unknown Bank";

  const [isAccountSelectorOpen, setIsAccountSelectorOpen] = useState(false);
  const [selectedAccountIds, setSelectedAccountIds] = useState<Key[]>([]);

  const sessionRef = useRef(currentSessionId);
  const bankRef = useRef(bankName);

  useEffect(() => {
    sessionRef.current = currentSessionId;
    bankRef.current = bankName;
  }, [currentSessionId, bankName]);

  const createNewSession = (title: string, contextData?: AdvisorContext) => {
    const newSession: ChatSession = {
      id: crypto.randomUUID(),
      title,
      messages: [],
      lastUpdated: new Date(),
      contextData,
    };
    setLocalSessions((prev) => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    return newSession.id;
  };

  const deleteSession = async (id: string) => {
    setLocalSessions((prev) => prev.filter((s) => s.id !== id));
    if (currentSessionId === id) {
      setCurrentSessionId(null);
    }
    try {
      await apiFetch(`/api/chat/sessions/${id}`, { method: "DELETE" }, true);
    } catch (e: unknown) {
      console.error(e);
    }
  };

  const [explanation, setExplanation] = useState<ExplanationState>({
    isOpen: false,
    isExplaining: false,
    contextType: null,
    rawPayload: null,
    aiExplanation: "",
    nextActions: [],
  });

  const triggerExplanation = (
    type: ExplanationContextType,
    data: ExplanationPayload | null,
  ) => {
    setExplanation((prev) => ({
      ...prev,
      isOpen: true,
      contextType: type,
      rawPayload: data,
      aiExplanation: "",
      nextActions: [],
    }));
  };

  const sendChatMessage = async (textToSend: string, sessionId?: string) => {
    if (!textToSend.trim()) return;
    const targetUrl = sessionId
      ? `/advisor?session=${encodeURIComponent(sessionId)}&prompt=${encodeURIComponent(textToSend.trim())}`
      : `/advisor?prompt=${encodeURIComponent(textToSend.trim())}`;
    router.push(targetUrl);
    return sessionId || "new";
  };

  const handleAiChartTrigger = async (
    type: TabType | string,
    customParams?: Partial<ToolParameters>,
  ) => {
    setIsGenerating(true);
    setIsChatOpen(true);

    const targetId = customParams?.bank_name_or_id || activeAccountId;

    let bankName = "ALL";
    if (targetId !== "ALL" && targetId) {
      const foundAccount = fetchedAccounts.find(
        (a) => a.account_id === targetId || a.bank_name === targetId,
      );
      bankName = foundAccount
        ? foundAccount.bank_name || foundAccount.provider_name || targetId
        : targetId;
    }

    const bankStr =
      bankName === "ALL" ? "all my accounts" : `my ${bankName} account`;
    let prompt = `Analyze ${type} for ${bankStr}.`;

    if (type === "expense_forecast")
      prompt = `Generate an expense forecast for the next ${customParams?.days || 30} days for ${bankStr}.`;
    else if (type === "balance_forecast")
      prompt = `Project my balance forecast for the next ${customParams?.days || 60} days for ${bankStr}.`;
    else if (type === "categorized")
      prompt = `Categorize my spending from ${customParams?.from_date || "2024-01-01"} to ${customParams?.to_date || new Date().toISOString().split("T")[0]} for ${bankStr}.`;
    else if (type === "historical_monthly")
      prompt = `Show my monthly historical expenses from ${customParams?.from_date || "2024-01-01"} to ${customParams?.to_date || new Date().toISOString().split("T")[0]} for ${bankStr}.`;
    else if (type === "health_radar")
      prompt = `Show my financial health radar for ${bankStr}.`;
    else if (type === "cash_flow_mixed")
      prompt = `Plot my cash flow for ${bankStr}.`;

    await sendChatMessage(prompt);
  };

  return (
    <BudAIContext.Provider
      value={{
        accounts: fetchedAccounts,
        activeAccountId,
        setActiveAccountId,
        chartConfig,
        setChartConfig,
        isGenerating,
        setIsGenerating,
        handleAiChartTrigger,
        userName,
        totalBalance,
        explanation,
        setExplanation,
        triggerExplanation,
        isChatOpen,
        setIsChatOpen,
        currentSessionId,
        setCurrentSessionId,
        sessions: localSessions,
        createNewSession,
        deleteSession,
        isChatLoading: false,
        chatMessages: [],
        sendChatMessage,
        thinkingProfile,
        isAccountSelectorOpen,
        setIsAccountSelectorOpen,
        selectedAccountIds,
        setSelectedAccountIds,
        snapshot,
        snapshotDays,
        setSnapshotDays: handleSetSnapshotDays,
        snapshotPage,
        setSnapshotPage,
        isSnapshotFetching,
      }}
    >
      {children}
      <Modal.Backdrop
        isOpen={isAccountSelectorOpen}
        onOpenChange={setIsAccountSelectorOpen}
        variant="blur"
      >
        <Modal.Container className="liquid-glass border-none shadow-2xl rounded-3xl">
          <Modal.Dialog>
            <Modal.Header>
              <div className="flex items-center gap-3">
                <div className="bg-primary/10 text-primary border border-primary/20 p-2 rounded-xl ">
                  <Wallet size={20} />
                </div>
                <h3 className="text-foreground font-black uppercase tracking-[0.4em] text-xs">
                  Account Resolution Required
                </h3>
              </div>
              <p className="text-muted-foreground text-[11px] font-medium uppercase tracking-widest mt-2">
                Multiple active accounts detected. Select target(s) for
                analysis.
              </p>
            </Modal.Header>
            <Modal.Body className="p-8 pt-4">
              <div className="flex flex-col gap-2">
                <Label className="text-[10px] font-black uppercase tracking-widest mb-1 block opacity-40">
                  Linked Institutions
                </Label>
                <Select className="w-full" variant="primary">
                  <Select.Trigger className="rounded-xl border border-white/10 bg-white/5 p-4 flex items-center justify-between hover:bg-white/10 transition-all">
                    <Select.Value className="text-xs font-bold uppercase tracking-tight">
                      {selectedAccountIds.length === 0
                        ? "Select accounts..."
                        : selectedAccountIds.length === fetchedAccounts.length
                          ? "All Accounts"
                          : `${selectedAccountIds.length} Accounts Selected`}
                    </Select.Value>
                    <Select.Indicator />
                  </Select.Trigger>
                  <Select.Popover className="liquid-glass border-white/10 rounded-2xl shadow-2xl">
                    <ListBox
                      selectionMode="multiple"
                      selectedKeys={new Set(selectedAccountIds)}
                      onSelectionChange={(keys: Selection) => {
                        if (keys === "all") {
                          setSelectedAccountIds(
                            fetchedAccounts.map((a) => a.account_id),
                          );
                        } else {
                          setSelectedAccountIds(Array.from(keys as Set<Key>));
                        }
                      }}
                      className="p-2"
                    >
                      {fetchedAccounts.map((acc) => (
                        <ListBox.Item
                          key={acc.account_id}
                          id={acc.account_id}
                          textValue={acc.bank_name || acc.provider_name}
                          className="rounded-xl p-3 hover:bg-white/5 transition-all border border-transparent hover:border-white/5"
                        >
                          <div className="flex flex-col gap-0.5">
                            <span className="text-[11px] font-black uppercase tracking-tight text-foreground">
                              {acc.bank_name || acc.provider_name}
                            </span>
                            <span className="text-[10px] text-muted-foreground font-mono italic tracking-tight">
                              {acc.account_number}
                            </span>
                          </div>
                        </ListBox.Item>
                      ))}
                    </ListBox>
                  </Select.Popover>
                </Select>
              </div>
            </Modal.Body>
            <Modal.Footer className="p-8 pt-0 flex gap-4">
              <Button
                variant="ghost"
                onPress={() => setIsAccountSelectorOpen(false)}
                className="flex-1 text-[11px] font-bold uppercase tracking-widest text-muted-foreground hover:text-white transition-all border-none h-12 rounded-xl"
              >
                Cancel
              </Button>
              <Button
                variant="secondary" className="flex-2 font-black uppercase tracking-widest text-[11px] h-12 rounded-xl transition-all "
                onPress={async () => {
                  if (selectedAccountIds.length > 0) {
                    setIsAccountSelectorOpen(false);
                    const names = selectedAccountIds
                      .map((id) => {
                        const acc = fetchedAccounts.find(
                          (a) => a.account_id === id,
                        );
                        return acc?.bank_name || acc?.provider_name || id;
                      })
                      .join(", ");
                    await sendChatMessage(
                      `Analyze these accounts: ${names} (IDs: ${selectedAccountIds.join(", ")})`,
                    );
                  }
                }}
              >
                Analyze
              </Button>
            </Modal.Footer>
          </Modal.Dialog>
        </Modal.Container>
      </Modal.Backdrop>
    </BudAIContext.Provider>
  );
};

export const useBudAI = () => {
  const context = useContext(BudAIContext);
  if (!context) throw new Error("useBudAI must be used within BudAIProvider");
  return context;
};
