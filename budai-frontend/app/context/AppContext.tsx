// AppContext.tsx
"use client";

import React, { createContext, useContext, useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import {
  Account,
  NativeChartConfig,
  TabType,
  ToolParameters,
  ExplanationState,
  ExplanationContextType,
  ExplanationPayload,
  LocalMessage,
} from "@/types";
import { buildChartConfig } from "@/app/(protected)/_utils/ChartBuilder";
import { apiFetch } from "@/lib/api";

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
  chatMessages: LocalMessage[];
  isChatLoading: boolean;
  sendChatMessage: (textToSend: string) => Promise<void>;
}

const BudAIContext = createContext<BudAIContextType | undefined>(undefined);

export const BudAIProvider = ({ children }: { children: React.ReactNode }) => {
  const router = useRouter();
  const [accounts, setAccounts] = useState<Account[]>([]);
  const [activeAccountId, setActiveAccountId] = useState<string | null>("ALL");
  const [chartConfig, setChartConfig] = useState<NativeChartConfig | null>(
    null,
  );
  const [isGenerating, setIsGenerating] = useState<boolean>(false);

  const [isChatOpen, setIsChatOpen] = useState<boolean>(false);
  const [chatMessages, setChatMessages] = useState<LocalMessage[]>([]);
  const [isChatLoading, setIsChatLoading] = useState<boolean>(false);

  const [userName] = useState<string>(
    typeof window !== "undefined"
      ? localStorage.getItem("budai_user_name") || "User"
      : "User",
  );

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
    const token = localStorage.getItem("budai_token");
    if (!token) return router.push("/login");
    apiFetch("/api/accounts/", {}, true)
      .then((res) => res.json())
      .then((data) => {
        if (data.accounts) setAccounts(data.accounts);
      })
      .catch(() => {});
  }, [router]);

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

  const sendChatMessage = async (textToSend: string) => {
    if (!textToSend.trim() || isChatLoading) return;

    setChatMessages((prev) => [
      ...prev,
      { role: "user", text: textToSend, timestamp: new Date() },
    ]);
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
        "/api/chat/",
        {
          method: "POST",
          body: JSON.stringify({
            input: textToSend,
            active_account_id: bankName,
            user_id: localStorage.getItem("budai_token"),
            chat_history: chatMessages.map(({ role, text }) => ({
              role,
              text,
            })),
          }),
        },
        true,
      );

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiText = "";

      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: "", timestamp: new Date() },
      ]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;
        aiText += decoder.decode(value, { stream: true });
        setChatMessages((prev) => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1].text = aiText;
          return newMessages;
        });
      }

      const triggerRegex =
        /\[TRIGGER_([A-Z_]+)_CHART(?:[:]([^\]:]+))?(?:[:]([^\]]+))?\]/g;
      let cleanedReply = aiText;
      let match;
      let foundTrigger = false;

      while ((match = triggerRegex.exec(aiText)) !== null) {
        const rawType = match[1].toLowerCase();
        const cacheId = match[2];

        let triggeredAction = "";
        if (rawType === "categorized") triggeredAction = "categorized_doughnut";
        else if (rawType === "balance") triggeredAction = "balance_forecast";
        else if (rawType === "expense") triggeredAction = "expense_forecast";
        else if (rawType === "cash_flow") triggeredAction = "cash_flow_mixed";
        else if (rawType === "health_radar") triggeredAction = "health_radar";
        else if (rawType.startsWith("historical")) triggeredAction = rawType;

        if (triggeredAction && cacheId) {
          foundTrigger = true;
          fetchCachedChart(triggeredAction, cacheId);
        }
        cleanedReply = cleanedReply.replace(match[0], "");
      }

      setChatMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1].text = cleanedReply.trim();
        return newMessages;
      });

      if (!foundTrigger) setIsGenerating(false);
    } catch (e) {
      console.error(e);
      setChatMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Engine Error.", timestamp: new Date() },
      ]);
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
    (sum, acc) => sum + (acc.balance ?? acc.account_balance ?? 0),
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
        chatMessages,
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
