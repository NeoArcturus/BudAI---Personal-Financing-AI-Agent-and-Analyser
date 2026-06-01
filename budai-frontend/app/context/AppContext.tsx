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
} from "@heroui/react";
import { useRouter } from "next/navigation";
import { useChat, UIMessage } from "@ai-sdk/react";
import { DefaultChatTransport, JSONValue } from "ai";
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
import { useAccounts, useChatSessions } from "@/lib/hooks";
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
}

const BudAIContext = createContext<BudAIContextType | undefined>(undefined);

export const BudAIProvider = ({
  children,
  initialSessions = [],
}: {
  children: React.ReactNode;
  initialAccounts?: Account[];
  initialSessions?: ChatSession[];
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
  }, []);

  const { data: fetchedAccounts = [] } = useAccounts();
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
    return fetchedAccounts.reduce((sum, acc) => sum + (acc.balance ?? 0), 0);
  }, [fetchedAccounts]);

  const fetchCachedChart = useCallback(
    async (type: string, cacheId: string) => {
      let toolName = "";
      if (type.includes("categorized")) toolName = "classify_financial_data";
      else if (type.includes("expense_forecast"))
        toolName = "generate_expense_forecast";
      else if (type.includes("balance_forecast"))
        toolName = "generate_financial_forecast";
      else if (type.includes("health")) toolName = "plot_health_radar";
      else if (type.includes("cash_flow")) toolName = "plot_cash_flow_mixed";
      else if (type.includes("historical")) toolName = "plot_expenses";

      try {
        setIsGenerating(true);
        const res = await apiFetch(
          "/api/media/execute",
          {
            method: "POST",
            body: JSON.stringify({
              tool_name: toolName,
              parameters: { bank_name_or_id: cacheId },
            }),
          },
          true,
        );
        const jsonRes = (await res.json()) as { data?: BankChartData[] };
        const newConfig = buildChartConfig(
          type,
          jsonRes.data || [],
          { bank_name_or_id: cacheId },
          "Targeted Analysis View",
        );
        if (newConfig) {
          setChartConfig(newConfig);
          router.push("/home");
        }
      } catch (e: unknown) {
        console.error(e);
      } finally {
        setIsGenerating(false);
      }
    },
    [router],
  );

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

  const chat = useChat({
    id: "budai-primary-chat",
    experimental_throttle: 50,
    transport: new DefaultChatTransport({
      api: getApiUrl("/api/chat/stream"),
      headers: () => ({
        Authorization: `Bearer ${getAuthToken()}`,
      }),
      prepareSendMessagesRequest: ({ messages, trigger, messageId }) => {
        return {
          body: {
            messages,
            trigger,
            messageId,
            session_id: sessionRef.current,
            active_account_id: bankRef.current,
          },
        };
      },
    }),
    async onToolCall({ toolCall }) {
      if (toolCall.toolName === "render_ui_chart") {
        const payload = toolCall as typeof toolCall & {
          args: { chart_type: string; cache_id: string };
        };
        if (payload.args && payload.args.chart_type && payload.args.cache_id) {
          fetchCachedChart(payload.args.chart_type, payload.args.cache_id);
        }
      }
      if (toolCall.toolName === "render_account_selector") {
        setIsAccountSelectorOpen(true);
      }
    },
    onData(dataPart) {
      if (dataPart && typeof dataPart === "object" && "data" in dataPart) {
        const part = dataPart as typeof dataPart & { data: JSONValue };
        if (Array.isArray(part.data)) {
          part.data.forEach((item) => {
            const latestData = item as typeof item & {
              type?: string;
              title?: string;
              profile_used?: string;
            };
            if (
              latestData.type === "session_title_update" &&
              latestData.title
            ) {
              setLocalSessions((prev) =>
                prev.map((s) =>
                  s.id === sessionRef.current
                    ? { ...s, title: latestData.title as string }
                    : s,
                ),
              );
            } else if (latestData.type === "global_refresh_signal") {
              queryClient.invalidateQueries();
            } else if (
              latestData.type === "thinking_context" &&
              latestData.profile_used
            ) {
              setThinkingProfile(latestData.profile_used as string);
            } else if (latestData.type === "trigger_account_selector") {
              setIsAccountSelectorOpen(true);
            }
          });
        }
      }
    },
    onError(error) {
      console.error("[BudAI Stream Error]:", error);
    },
  });

  const { messages, status, setMessages, sendMessage } = chat;

  const isLoading = status === "streaming" || status === "submitted";

  const localSessionsRef = useRef(localSessions);

  useEffect(() => {
    localSessionsRef.current = localSessions;
  }, [localSessions]);

  useEffect(() => {
    if (!currentSessionId || isLoading) return;

    const currentSession = localSessionsRef.current.find(
      (s) => s.id === currentSessionId,
    );
    if (!currentSession) return;

    if (currentSession.messages.length === 0) {
      apiFetch(`/api/chat/sessions/${currentSessionId}`, {}, true)
        .then((res) => res.json())
        .then((data: SessionResponse) => {
          if (
            data &&
            typeof data === "object" &&
            "messages" in data &&
            Array.isArray((data as SessionResponse).messages)
          ) {
            const typedData = data as SessionResponse;
            const serverMsgs: UIMessage[] = typedData.messages.map(
              (m: ServerMessage) =>
                ({
                  id: m.id || crypto.randomUUID(),
                  role: m.role,
                  content: m.content,
                  parts: [{ type: "text", text: m.content }],
                  createdAt: new Date(m.timestamp),
                  status: "ready",
                }) as UIMessage,
            );

            if (!isLoading) {
              setMessages(serverMsgs);
              setLocalSessions((prev) =>
                prev.map((s) =>
                  s.id === currentSessionId
                    ? { ...s, messages: serverMsgs }
                    : s,
                ),
              );
            }
          }
        })
        .catch(() => {});
    } else {
      setMessages((current) => {
        if (isLoading) return current;
        if (
          current.length > 0 &&
          current[current.length - 1].role === "assistant"
        ) {
          return current;
        }
        return currentSession.messages;
      });
    }
  }, [currentSessionId, isLoading, setMessages]);

  useEffect(() => {
    if (!currentSessionId || messages.length === 0) return;

    setLocalSessions((prev) => {
      const session = prev.find((s) => s.id === currentSessionId);
      if (!session) return prev;

      const isSame =
        session.messages.length === messages.length &&
        session.messages.every((m, i) => m.id === messages[i]?.id);

      if (isSame) return prev;

      return prev.map((s) =>
        s.id === currentSessionId
          ? { ...s, messages, lastUpdated: new Date() }
          : s,
      );
    });
  }, [messages, currentSessionId]);

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
    setMessages([]);
    return newSession.id;
  };

  const deleteSession = async (id: string) => {
    setLocalSessions((prev) => prev.filter((s) => s.id !== id));
    if (currentSessionId === id) {
      setCurrentSessionId(null);
      setMessages([]);
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
    if (!textToSend.trim() || isLoading) return;

    let targetId = sessionId || currentSessionId;
    if (!targetId) {
      targetId = createNewSession("New Conversation");
    } else if (targetId !== currentSessionId) {
      setCurrentSessionId(targetId);
    }

    await sendMessage({
      id: crypto.randomUUID(),
      role: "user",
      parts: [{ type: "text", text: textToSend }],
    });

    setLocalSessions((prev) =>
      prev.map((s) =>
        s.id === targetId
          ? {
              ...s,
              title:
                s.messages.length <= 2
                  ? textToSend.length > 30
                    ? textToSend.slice(0, 30) + "..."
                    : textToSend
                  : s.title,
            }
          : s,
      ),
    );

    return targetId;
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
        isChatLoading: isLoading,
        chatMessages: messages || [],
        sendChatMessage,
        thinkingProfile,
        isAccountSelectorOpen,
        setIsAccountSelectorOpen,
        selectedAccountIds,
        setSelectedAccountIds,
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
                <div className="p-2 rounded-xl bg-primary/10 text-primary border border-primary/20">
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
                className="flex-2 bg-primary text-primary-foreground font-black uppercase tracking-widest text-[11px] h-12 rounded-xl shadow-[0_0_20px_rgba(0,127,255,0.3)] hover:shadow-[0_0_30px_rgba(0,127,255,0.5)] transition-all border-none"
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
