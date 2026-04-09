"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Loader2,
  Sparkles,
  TrendingUp,
  Activity,
  BarChart2,
} from "lucide-react";
import { ChatMessage, Account, TabType } from "@/types";
import { apiFetch } from "@/lib/api";

interface BudAIChatProps {
  onAiAction: (
    type: TabType | string,
    customTitle?: string,
    aiTargetId?: string,
    extraParam?: string,
  ) => void;
  activeAccountId: string | null;
  accounts: Account[];
}

export default function BudAIChat({
  onAiAction,
  activeAccountId,
  accounts,
}: BudAIChatProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState<string>("");
  const [loading, setLoading] = useState<boolean>(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (scrollRef.current) {
      scrollRef.current.scrollTop = scrollRef.current.scrollHeight;
    }
  }, [messages]);

  const handleSend = async (
    overrideText?: string,
    actionType?: TabType | string,
  ) => {
    const textToSend = overrideText || input;
    if (!textToSend.trim() || loading) return;

    setMessages((prev) => [...prev, { role: "user", text: textToSend }]);
    setInput("");
    setLoading(true);

    const activeAccount = accounts.find(
      (a) =>
        a.account_id === activeAccountId ||
        a.truelayer_account_id === activeAccountId,
    );

    const bankName =
      activeAccountId === "ALL"
        ? "ALL"
        : activeAccount
          ? activeAccount.bank_name ||
            activeAccount.provider_name ||
            "Unknown Bank"
          : "Unknown Bank";

    try {
      const response = await apiFetch(
        "/api/chat/",
        {
          method: "POST",
          body: JSON.stringify({
            input: textToSend,
            active_account_id: bankName,
            user_id: localStorage.getItem("budai_token"),
            chat_history: messages,
          }),
        },
        true,
      );

      if (!response.body) throw new Error("No response body");

      const reader = response.body.getReader();
      const decoder = new TextDecoder();
      let aiText = "";

      setMessages((prev) => [...prev, { role: "assistant", text: "" }]);

      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const chunk = decoder.decode(value, { stream: true });
        aiText += chunk;

        setMessages((prev) => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1].text = aiText;
          return newMessages;
        });
      }

      const triggerRegex =
        /\[TRIGGER_([A-Z_]+)_CHART(?:[:]([^\]:]+))?(?:[:]([^\]]+))?\]/g;
      let cleanedReply = aiText;
      let match;

      while ((match = triggerRegex.exec(aiText)) !== null) {
        const rawType = match[1].toLowerCase();
        const targetId = match[2];
        const extraParam = match[3];

        let triggeredAction = "";

        if (rawType === "categorized") triggeredAction = "categorized_doughnut";
        else if (rawType === "balance") triggeredAction = "balance_forecast";
        else if (rawType === "expense") triggeredAction = "expense_forecast";
        else if (rawType === "cash_flow") triggeredAction = "cash_flow_mixed";
        else if (rawType === "health_radar") triggeredAction = "health_radar";
        else if (rawType.startsWith("historical")) {
          const lowerText = aiText.toLowerCase();
          let timeType = "monthly";
          if (lowerText.includes("daily")) timeType = "daily";
          else if (lowerText.includes("weekly")) timeType = "weekly";
          triggeredAction = `historical_${timeType}`;
        }

        if (triggeredAction) {
          onAiAction(triggeredAction, undefined, targetId, extraParam);
        }
        cleanedReply = cleanedReply.replace(match[0], "");
      }

      setMessages((prev) => {
        const newMessages = [...prev];
        newMessages[newMessages.length - 1].text = cleanedReply.trim();
        return newMessages;
      });

      if (actionType && !aiText.includes("[TRIGGER_")) {
        onAiAction(actionType);
      }
    } catch (e: unknown) {
      console.log(e);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Critical Engine Error." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col h-screen bg-[#132017] border-l border-[#1A2D21] relative overflow-hidden shadow-2xl">
      <div className="p-4 border-b border-[#1A2D21] bg-[#0A120D] shrink-0 flex items-center justify-between z-10">
        <div className="flex items-center gap-3">
          <div className="w-8 h-8 rounded-full bg-[#69F0AE]/20 flex items-center justify-center border border-[#69F0AE]/30">
            <Sparkles className="text-[#69F0AE]" size={16} />
          </div>
          <div>
            <h2 className="text-sm font-bold tracking-widest text-white uppercase">
              BudAI
            </h2>
            <p className="text-[10px] text-[#69F0AE] font-mono tracking-wider">
              Financial Intelligence
            </p>
          </div>
        </div>
      </div>

      <div ref={scrollRef} className="flex-1 min-h-0 overflow-y-auto p-4">
        <div className="flex flex-col gap-4 pb-4">
          {messages.length === 0 ? (
            <div className="flex flex-col items-center justify-center h-full text-center p-6 mt-12">
              <div className="w-16 h-16 rounded-full bg-[#69F0AE]/10 flex items-center justify-center mb-6">
                <Sparkles className="text-[#69F0AE] w-8 h-8" />
              </div>
              <h3 className="text-xl font-bold text-white mb-2 font-mono">
                Hello, I am BudAI.
              </h3>
              <p className="text-sm text-slate-400 mb-8 max-w-xs">
                I can analyze your spending, forecast your balances, and track
                your financial health. Ask me anything.
              </p>

              <div className="flex flex-col gap-2 w-full max-w-sm">
                <button
                  onClick={() =>
                    handleSend("Categorize my expenses for the last 30 days")
                  }
                  className="flex items-center gap-3 p-3 bg-[#1A2D21]/50 border border-[#1A2D21] rounded-xl text-left hover:bg-[#1A2D21] hover:border-[#69F0AE]/30 transition-all group"
                >
                  <BarChart2 className="text-[#69F0AE] group-hover:scale-110 transition-transform w-5 h-5 shrink-0" />
                  <span className="text-sm text-slate-300 font-medium">
                    Analyze my spending
                  </span>
                </button>
                <button
                  onClick={() =>
                    handleSend(
                      "Predict my account balances for the next 60 days",
                    )
                  }
                  className="flex items-center gap-3 p-3 bg-[#1A2D21]/50 border border-[#1A2D21] rounded-xl text-left hover:bg-[#1A2D21] hover:border-[#69F0AE]/30 transition-all group"
                >
                  <TrendingUp className="text-[#69F0AE] group-hover:scale-110 transition-transform w-5 h-5 shrink-0" />
                  <span className="text-sm text-slate-300 font-medium">
                    Forecast my balance
                  </span>
                </button>
                <button
                  onClick={() =>
                    handleSend("Show me my financial health radar")
                  }
                  className="flex items-center gap-3 p-3 bg-[#1A2D21]/50 border border-[#1A2D21] rounded-xl text-left hover:bg-[#1A2D21] hover:border-[#69F0AE]/30 transition-all group"
                >
                  <Activity className="text-[#69F0AE] group-hover:scale-110 transition-transform w-5 h-5 shrink-0" />
                  <span className="text-sm text-slate-300 font-medium">
                    Check financial health
                  </span>
                </button>
              </div>
            </div>
          ) : (
            messages.map((msg, i) => (
              <div
                key={i}
                className={`flex flex-col gap-1 ${msg.role === "user" ? "items-end" : "items-start"}`}
              >
                <span className="text-[10px] font-mono tracking-wider text-[#69F0AE]/60 uppercase">
                  {msg.role === "user" ? "You" : "BudAI"}
                </span>
                <div
                  className={`max-w-[85%] rounded-xl px-4 py-2 text-sm leading-relaxed ${
                    msg.role === "user"
                      ? "bg-[#69F0AE]/20 text-white border border-[#69F0AE]/30"
                      : "bg-[#0A120D] text-white/90 border border-[#1A2D21]"
                  }`}
                >
                  {msg.text || (
                    <Loader2 className="w-3 h-3 animate-spin text-[#69F0AE]" />
                  )}
                </div>
              </div>
            ))
          )}
        </div>
      </div>

      <div className="p-4 bg-[#0A120D] border-t border-[#1A2D21] shrink-0">
        <div className="relative flex items-center gap-2 bg-[#132017] border border-[#233A2B] rounded-2xl px-4 py-3 focus-within:border-[#69F0AE]/50 focus-within:shadow-[0_0_0_1px_rgba(105,240,174,0.15)] transition-all duration-200">
          <span className="text-[#69F0AE]/40 font-mono text-xs shrink-0 select-none">
            ›_
          </span>
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Command BudAI..."
            className="flex-1 bg-transparent border-none outline-none text-sm text-white placeholder:text-[#69F0AE]/25 font-mono caret-[#69F0AE]"
          />
          <button
            onClick={() => handleSend()}
            disabled={loading || !input.trim()}
            className="shrink-0 w-8 h-8 flex items-center justify-center rounded-lg bg-[#69F0AE]/10 border border-[#69F0AE]/20 hover:bg-[#69F0AE]/20 hover:border-[#69F0AE]/40 disabled:opacity-20 disabled:cursor-not-allowed transition-all duration-150 group"
          >
            {loading ? (
              <Loader2 className="w-3.5 h-3.5 animate-spin text-[#69F0AE]" />
            ) : (
              <Send
                size={13}
                className="text-[#69F0AE] group-hover:translate-x-0.5 group-hover:-translate-y-0.5 transition-transform duration-150"
              />
            )}
          </button>
        </div>
        <p className="text-[9px] text-[#69F0AE]/20 font-mono text-center mt-2 tracking-widest uppercase">
          BudAI may make mistakes · Verify important figures
        </p>
      </div>
    </div>
  );
}
