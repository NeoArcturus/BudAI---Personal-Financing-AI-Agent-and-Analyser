"use client";

import React, { useState, useRef, useEffect } from "react";
import { Send, Loader2, Sparkles, TrendingUp, ShieldCheck } from "lucide-react";
import { ChatMessage, Account, TabType } from "@/types";
import ReactMarkdown from "react-markdown";

interface BudAIChatProps {
  onAiAction: (type: TabType) => void;
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
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages]);

  const handleSend = async (overrideText?: string, actionType?: TabType) => {
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
    const bankName = activeAccount
      ? activeAccount.bank_name || activeAccount.provider_name
      : "Unknown Bank";

    console.log("Active bank:", bankName);
    const token = localStorage.getItem("budai_token") || "";

    try {
      const response = await fetch("http://localhost:8080/api/chat/", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
          Authorization: `Bearer ${token}`,
        },
        body: JSON.stringify({
          input: textToSend,
          active_account_id: activeAccountId,
          user_id: localStorage.getItem("budai_token"),
          chat_history: messages,
        }),
      });

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

        // Strip the secret trigger tags out in real-time so they never appear in the UI
        const cleanText = aiText
          .replace("[TRIGGER_EXPENSE_CHART]", "")
          .replace("[TRIGGER_CATEGORIZED_CHART]", "")
          .replace("[TRIGGER_BALANCE_CHART]", "")
          .replace("[TRIGGER_HISTORICAL_CHART]", "");

        setMessages((prev) => {
          const newMessages = [...prev];
          newMessages[newMessages.length - 1].text = cleanText;
          return newMessages;
        });
      }

      // ---------------------------------------------------------
      // BULLETPROOF CHART TRIGGER LOGIC
      // ---------------------------------------------------------
      // We check the RAW 'aiText' (which still contains the tags)
      if (aiText.includes("[TRIGGER_EXPENSE_CHART]")) {
        onAiAction("expense_forecast");
      } else if (aiText.includes("[TRIGGER_CATEGORIZED_CHART]")) {
        onAiAction("categorized");
      } else if (aiText.includes("[TRIGGER_BALANCE_CHART]")) {
        onAiAction("balance_forecast");
      } else if (aiText.includes("[TRIGGER_HISTORICAL_CHART]")) {
        const lowerText = aiText.toLowerCase();
        let timeType = "monthly";

        if (lowerText.includes("daily")) timeType = "daily";
        else if (lowerText.includes("weekly")) timeType = "weekly";

        onAiAction(`historical_${timeType}` as TabType);
      } else if (actionType) {
        onAiAction(actionType);
      }
    } catch (e) {
      console.error("Error during AI communication:", e);
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Critical Engine Error." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="flex flex-col w-full h-full bg-[#161B22] border border-slate-800 rounded-3xl overflow-hidden shadow-2xl">
      <div className="p-4 border-b border-slate-800 bg-[#1c2128] flex items-center shrink-0">
        <span className="font-bold text-xs tracking-widest text-[#00FFAA] flex items-center gap-2">
          <Sparkles size={14} /> BUDAI CHAT
        </span>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide"
      >
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6">
            <h2 className="text-xl font-bold text-white">
              Ask BudAI to analyze your finances and generate insights,
              forecasts, and visualizations!
            </h2>
            <div className="grid grid-cols-2 gap-3 w-full max-w-sm">
              <button
                onClick={() =>
                  handleSend("Classify my recent transactions.", "categorized")
                }
                className="bg-[#0D1117] border border-slate-800 p-3 rounded-xl hover:border-[#00FFAA]/50 transition-all text-left flex items-center gap-2 text-xs text-slate-300 group"
              >
                <ShieldCheck
                  size={16}
                  className="text-[#00FFAA] group-hover:scale-110"
                />{" "}
                Categorize Data
              </button>
              <button
                onClick={() =>
                  handleSend(
                    "Generate a 30-day balance forecast.",
                    "balance_forecast",
                  )
                }
                className="bg-[#0D1117] border border-slate-800 p-3 rounded-xl hover:border-[#00FFAA]/50 transition-all text-left flex items-center gap-2 text-xs text-slate-300 group"
              >
                <TrendingUp
                  size={16}
                  className="text-[#00FFAA] group-hover:scale-110"
                />{" "}
                Run Forecast
              </button>
            </div>
          </div>
        ) : (
          messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] p-4 rounded-2xl text-sm leading-relaxed ${m.role === "user" ? "bg-[#00FFAA] text-black font-semibold rounded-br-none shadow-lg" : "bg-[#1c2128] text-slate-200 border border-slate-700/50 rounded-bl-none shadow-md"}`}
              >
                {m.role === "assistant" ? (
                  <div className="prose prose-invert max-w-none prose-sm prose-p:leading-relaxed prose-pre:bg-[#0D1117] prose-li:marker:text-[#00FFAA]">
                    <ReactMarkdown>{m.text}</ReactMarkdown>
                  </div>
                ) : (
                  m.text
                )}
              </div>
            </div>
          ))
        )}
        {loading && (
          <div className="flex items-center gap-2 text-[#00FFAA] text-xs font-mono px-2">
            <Loader2 className="w-4 h-4 animate-spin" /> BudAI is thinking...
          </div>
        )}
      </div>

      <div className="p-4 bg-[#1c2128] border-t border-slate-800 shrink-0">
        <div className="relative w-full">
          <input
            value={input}
            onChange={(e) => setInput(e.target.value)}
            onKeyDown={(e) => e.key === "Enter" && handleSend()}
            placeholder="Command BudAI..."
            className="w-full bg-[#0D1117] border border-slate-700 rounded-xl py-3 pl-4 pr-12 text-sm text-white focus:border-[#00FFAA] outline-none transition-all"
          />
          <button
            onClick={() => handleSend()}
            className="absolute right-2 top-1.5 p-1.5 bg-[#00FFAA] text-black rounded-lg hover:scale-105 transition-transform"
          >
            <Send size={16} />
          </button>
        </div>
      </div>
    </div>
  );
}
