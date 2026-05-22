// AppContext.tsx
"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
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
  LocalMessage,
} from "@/types";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { apiFetch } from "@/lib/api";

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
  messages: LocalMessage[];
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
  sendChatMessage: (textToSend: string, sessionId?: string) => Promise<void>;
}

const BudAIContext = createContext<BudAIContextType | undefined>(undefined);

export const BudAIProvider = ({
  children,
  initialAccounts = [],
  initialSessions = [],
}: {
  children: React.ReactNode;
  initialAccounts?: Account[];
  initialSessions?: ChatSession[];
}) => {
  const router = useRouter();
  const [accounts, setAccounts] = useState<Account[]>(initialAccounts);
  const [activeAccountId, setActiveAccountId] = useState<string | null>("ALL");
  const [chartConfig, setChartConfig] = useState<NativeChartConfig | null>(
    null,
  );
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const [isChatOpen, setIsChatOpen] = useState<boolean>(false);
  const [sessions, setSessions] = useState<ChatSession[]>(initialSessions);
  const [currentSessionId, setCurrentSessionId] = useState<string | null>(null);
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);

  const [userName] = useState<string>(
    typeof window !== "undefined"
      ? localStorage.getItem("budai_user_name") || "User"
      : "User",
  );

  useEffect(() => {
    if (!currentSessionId) return;

    const currentSession = sessions.find((s) => s.id === currentSessionId);
    if (currentSession && currentSession.messages.length === 0) {
      apiFetch(`/api/chat/sessions/${currentSessionId}`, {}, true)
        .then((res) => res.json())
        .then((data) => {
          if (data.messages) {
            setSessions((prev) =>
              prev.map((s) =>
                s.id === currentSessionId
                  ? {
                      ...s,
                      messages: data.messages.map(
                        (m: {
                          role: "user" | "assistant";
                          content: string;
                          timestamp: string;
                        }) => ({
                          role: m.role,
                          text: m.content,
                          timestamp: new Date(m.timestamp),
                        }),
                      ),
                    }
                  : s,
              ),
            );
          }
        })
        .catch((err) => console.error("Failed to fetch session details", err));
    }
  }, [currentSessionId, sessions]);

  const createNewSession = (title: string, contextData?: AdvisorContext) => {
    const newSession: ChatSession = {
      id: crypto.randomUUID(),
      title,
      messages: [],
      lastUpdated: new Date(),
      contextData,
    };
    setSessions((prev) => [newSession, ...prev]);
    setCurrentSessionId(newSession.id);
    return newSession.id;
  };

  const deleteSession = (id: string) => {
    setSessions((prev) => prev.filter((s) => s.id !== id));
    if (currentSessionId === id) setCurrentSessionId(null);
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

  useEffect(() => {
    const token = typeof window !== "undefined" ? localStorage.getItem("budai_token") : null;
    if (!token && typeof window !== "undefined") return router.push("/login");

    if (initialAccounts.length === 0) {
        apiFetch("/api/accounts", {}, true)
          .then((res) => res.json())
          .then((data) => {
            if (data.accounts) setAccounts(data.accounts);
          })
          .catch(() => {});
    }

    if (initialSessions.length === 0) {
        apiFetch("/api/chat/sessions", {}, true)
          .then((res) => res.json())
          .then((data) => {
            if (Array.isArray(data)) {
              const mappedSessions: ChatSession[] = data.map((s) => ({
                id: s.session_id,
                title: s.title,
                messages: [],
                lastUpdated: new Date(s.last_updated),
                contextData: s.context_data,
              }));
              setSessions(mappedSessions);
            }
          })
          .catch((err) => console.error("Failed to fetch sessions", err));
    }
  }, [router, initialAccounts, initialSessions]);

  const fetchCachedChart = async (type: string, cacheId: string) => {
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
      const jsonRes = await res.json();
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
    } catch (e) {
      console.error(e);
    } finally {
      setIsGenerating(false);
    }
  };

  const sendChatMessage = async (textToSend: string, sessionId?: string) => {
    if (!textToSend.trim() || isChatLoading) return;

    let targetId = sessionId || currentSessionId;
    if (!targetId) {
      targetId = createNewSession("New Conversation");
    }

    const currentSession = sessions.find((s) => s.id === targetId);
    const history = currentSession?.messages || [];

    const userMsg: LocalMessage = {
      role: "user",
      text: textToSend,
      timestamp: new Date(),
    };
    setSessions((prev) =>
      prev.map((s) =>
        s.id === targetId
          ? {
              ...s,
              messages: [...s.messages, userMsg],
              lastUpdated: new Date(),
            }
          : s,
      ),
    );

    setIsChatLoading(true);

    const activeAccount = accounts.find(
      (a) => a.account_id === activeAccountId,
    );
    const bankName =
      activeAccountId === "ALL"
        ? "ALL"
        : activeAccount?.bank_name || "Unknown Bank";

    try {
      const response = await apiFetch(
        "/api/chat",
        {
          method: "POST",
          body: JSON.stringify({
            input: textToSend,
            active_account_id: bankName,
            session_id: targetId,
            chat_history: history.map(({ role, text }) => ({ role, text })),
            context_data: currentSession?.contextData,
          }),
        },
        true,
      );

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiText = "";

      setSessions((prev) =>
        prev.map((s) =>
          s.id === targetId
            ? {
                ...s,
                messages: [
                  ...s.messages,
                  { role: "assistant", text: "", timestamp: new Date() },
                ],
              }
            : s,
        ),
      );

      const sessionIdRegex = /^\[SESSION_ID:([a-zA-Z0-9-]+)\]/;

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        let chunk = decoder.decode(value, { stream: true });

        const sessionMatch = chunk.match(sessionIdRegex);
        if (sessionMatch) {
          const newSessionId = sessionMatch[1];
          setCurrentSessionId(newSessionId);

          if (targetId !== newSessionId) {
            setSessions((prev) =>
              prev.map((s) =>
                s.id === targetId ? { ...s, id: newSessionId } : s,
              ),
            );
            
            // Promote targetId to the new one and update URL if on advisor page
            targetId = newSessionId;
            if (typeof window !== "undefined" && window.location.pathname === "/advisor") {
              router.replace(`/advisor?session=${newSessionId}`);
            }
          }
          chunk = chunk.replace(sessionIdRegex, "");
        }

        aiText += chunk;

        setSessions((prev) =>
          prev.map((s) =>
            s.id === targetId
              ? {
                  ...s,
                  messages: s.messages.map((m, idx) =>
                    idx === s.messages.length - 1 ? { ...m, text: aiText } : m,
                  ),
                }
              : s,
          ),
        );
      }

      const triggerRegex =
        /\[TRIGGER_([A-Z_]+)_CHART(?:[:]([^\]:]+))?(?:[:]([^\]]+))?\]/g;
      let cleanedReply = aiText;
      let match;
      let foundTrigger = false;

      while ((match = triggerRegex.exec(aiText)) !== null) {
        const typeSlug = match[1].toLowerCase();
        const cacheId = match[2];

        let triggeredAction = "";
        if (typeSlug === "categorized")
          triggeredAction = "categorized_doughnut";
        else if (typeSlug === "balance_forecast")
          triggeredAction = "balance_forecast";
        else if (typeSlug === "expense") triggeredAction = "expense_forecast";
        else if (typeSlug === "cash_flow") triggeredAction = "cash_flow_mixed";
        else if (typeSlug === "health_radar") triggeredAction = "health_radar";
        else if (typeSlug.startsWith("historical")) triggeredAction = typeSlug;

        if (triggeredAction && cacheId) {
          foundTrigger = true;
          fetchCachedChart(triggeredAction, cacheId);
        }
        cleanedReply = cleanedReply.replace(match[0], "");
      }

      setSessions((prev) =>
        prev.map((s) =>
          s.id === targetId
            ? {
                ...s,
                messages: s.messages.map((m, idx) =>
                  idx === s.messages.length - 1
                    ? { ...m, text: cleanedReply.trim() }
                    : m,
                ),
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

      if (!foundTrigger) setIsGenerating(false);
    } catch (e) {
      console.error(e);
      setSessions((prev) =>
        prev.map((s) =>
          s.id === targetId
            ? {
                ...s,
                messages: [
                  ...s.messages,
                  {
                    role: "assistant",
                    text: "Analysis service unavailable. Please try again later.",
                    timestamp: new Date(),
                  },
                ],
              }
            : s,
        ),
      );
      setIsGenerating(false);
    } finally {
      setIsChatLoading(false);
    }
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
      const foundAccount = accounts.find(
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

  const totalBalance = accounts.reduce(
    (sum, acc) => sum + (acc.balance ?? 0),
    0,
  );

  return (
    <BudAIContext.Provider
      value={{
        accounts,
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
        sessions,
        createNewSession,
        deleteSession,
        isChatLoading,
        sendChatMessage,
      }}
    >
      {children}
    </BudAIContext.Provider>
  );
};

export const useBudAI = () => {
  const context = useContext(BudAIContext);
  if (!context) throw new Error("useBudAI must be used within BudAIProvider");
  return context;
};
