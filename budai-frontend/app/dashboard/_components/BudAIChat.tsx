"use client";

import React, { useState, useRef, useEffect } from "react";
import {
  Send,
  Loader2,
  Sparkles,
  TrendingUp,
  ShieldCheck,
  LineChart,
  Globe,
} from "lucide-react";
import { ChatMessage } from "@/types";
import ReactMarkdown from "react-markdown";

type TabType = "raw" | "categorized" | "balance_forecast" | "expense_forecast";

export default function BudAIChat({
  onAiAction,
  activeAccountId,
}: {
  onAiAction: (type: TabType) => void;
  activeAccountId: string | null;
}) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const scrollRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    scrollRef.current?.scrollTo(0, scrollRef.current.scrollHeight);
  }, [messages]);

  const handleSend = async (overrideText?: string, actionType?: TabType) => {
    const textToSend = overrideText || input;
    if (!textToSend.trim() || loading) return;

    // Display the clean message on the frontend UI
    setMessages((prev) => [...prev, { role: "user", text: textToSend }]);
    setInput("");
    setLoading(true);

    // Contextually inject the activeAccountId for the backend LLM without showing it to the user
    const apiInput = activeAccountId
      ? `[System Note: The user has currently selected account_id: ${activeAccountId} on their dashboard] \n\n${textToSend}`
      : textToSend;

    try {
      const response = await fetch("http://localhost:8080/api/chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ input: apiInput, chat_history: messages }),
      });
      const data = await response.json();

      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: data.output },
      ]);

      const lowerOutput = data.output.toLowerCase();

      const triggeredCategory =
        lowerOutput.includes("classif") ||
        lowerOutput.includes("categoriz") ||
        lowerOutput.includes("spending broke down") ||
        lowerOutput.includes("breakdown");

      const triggeredExpense =
        lowerOutput.includes("expense forecast") ||
        lowerOutput.includes("quantitative convergence report");

      const triggeredForecast =
        lowerOutput.includes("forecast") ||
        lowerOutput.includes("stochastic") ||
        lowerOutput.includes("simulation");

      if (triggeredCategory) {
        onAiAction("categorized");
      } else if (triggeredExpense) {
        onAiAction("expense_forecast");
      } else if (triggeredForecast) {
        onAiAction("balance_forecast");
      } else if (actionType) {
        onAiAction(actionType);
      }
    } catch {
      setMessages((prev) => [
        ...prev,
        { role: "assistant", text: "Critical Engine Error." },
      ]);
    } finally {
      setLoading(false);
    }
  };

  const quickActions = [
    {
      label: "Categorize Data",
      icon: <ShieldCheck size={16} />,
      prompt: "Classify my financial data from 2026-01-01 to 2026-03-03.",
      type: "categorized",
    },
    {
      label: "Run Forecast",
      icon: <TrendingUp size={16} />,
      prompt: "Generate a financial forecast for the next 30 days.",
      type: "balance_forecast",
    },
    {
      label: "Analyze Spending",
      icon: <LineChart size={16} />,
      prompt: "Find the total spent for all categories.",
    },
    {
      label: "Market News",
      icon: <Globe size={16} />,
      prompt: "Analyze current commodity markets.",
    },
  ];

  return (
    <div className="flex flex-col w-full h-full bg-[#161B22] border border-slate-800 rounded-3xl overflow-hidden shadow-2xl">
      <div className="p-4 border-b border-slate-800 bg-[#1c2128] flex items-center shrink-0">
        <span className="font-bold text-xs tracking-widest text-[#00FFAA] flex items-center gap-2">
          <Sparkles size={14} /> AGENTIC UI
        </span>
      </div>

      <div
        ref={scrollRef}
        className="flex-1 overflow-y-auto p-6 space-y-6 scrollbar-hide"
      >
        {messages.length === 0 ? (
          <div className="h-full flex flex-col items-center justify-center text-center space-y-6">
            <h2 className="text-xl font-bold text-white">
              How can I assist your financial strategy?
            </h2>
            <div className="grid grid-cols-2 gap-3 w-full max-w-sm">
              {quickActions.map((action, i) => (
                <button
                  key={i}
                  onClick={() =>
                    handleSend(
                      action.prompt,
                      action.type as TabType | undefined,
                    )
                  }
                  className="bg-[#0D1117] border border-slate-800 p-3 rounded-xl hover:border-[#00FFAA]/50 transition-all text-left flex items-center gap-2 text-xs text-slate-300 group shadow-sm"
                >
                  <div className="text-[#00FFAA] group-hover:scale-110 transition-transform">
                    {action.icon}
                  </div>
                  {action.label}
                </button>
              ))}
            </div>
          </div>
        ) : (
          messages.map((m, i) => (
            <div
              key={i}
              className={`flex ${m.role === "user" ? "justify-end" : "justify-start"}`}
            >
              <div
                className={`max-w-[85%] p-4 rounded-2xl text-sm leading-relaxed ${
                  m.role === "user"
                    ? "bg-[#00FFAA] text-black font-semibold rounded-br-none shadow-lg"
                    : "bg-[#1c2128] text-slate-200 border border-slate-700/50 rounded-bl-none shadow-md"
                }`}
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
            placeholder="Query forecasting, categorization, or market analysis..."
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
