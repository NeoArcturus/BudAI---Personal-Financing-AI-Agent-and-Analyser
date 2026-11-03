"use client";

import React, { useState, useRef, useEffect } from "react";
import { Send, Square, Sparkles, CornerDownLeft } from "lucide-react";
import { Button, TextArea } from "@heroui/react";
import { motion } from "framer-motion";
import { cn } from "@/lib/utils";

export type VerbosityMode = "concise" | "balanced" | "deep_dive";

interface ChatInputAreaProps {
  status: string;
  stop: () => void;
  onSend: (text: string, verbosity: VerbosityMode) => void;
  isLocked?: boolean;
  verbosity: VerbosityMode;
  onVerbosityChange: (mode: VerbosityMode) => void;
}

const VERBOSITY_CONFIG: Record<
  VerbosityMode,
  { label: string; description: string }
> = {
  concise: {
    label: "Concise",
    description: "Brief numerical takeaways & direct answers",
  },
  balanced: {
    label: "Balanced",
    description: "Standard advisory synthesis with core context",
  },
  deep_dive: {
    label: "Deep Dive",
    description: "Detailed analysis and spending breakdown",
  },
};

export const ChatInputArea: React.FC<ChatInputAreaProps> = ({
  status,
  stop,
  onSend,
  isLocked = false,
  verbosity,
  onVerbosityChange,
}) => {
  const [input, setInput] = useState("");
  const [isMac, setIsMac] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  useEffect(() => {
    setIsMac(/Mac|iPod|iPhone|iPad/.test(navigator.platform || navigator.userAgent));
  }, []);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        180,
      )}px`;
    }
  }, [input]);

  const handleSend = () => {
    if (
      !input.trim() ||
      status === "streaming" ||
      status === "submitted" ||
      isLocked
    )
      return;
    onSend(input.trim(), verbosity);
    setInput("");
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  const isGenerating = status === "streaming" || status === "submitted";

  return (
    <motion.div
      initial={{ opacity: 0, y: 10 }}
      animate={{ opacity: 1, y: 0 }}
      transition={{ duration: 0.3 }}
      className={cn(
        "w-full rounded-2xl border transition-all duration-300 shadow-2xl backdrop-blur-2xl bg-content1/50",
        isLocked
          ? "border-danger/30 opacity-60 pointer-events-none bg-[repeating-linear-gradient(45deg,transparent,transparent_10px,rgba(255,255,255,0.03)_10px,rgba(255,255,255,0.03)_20px)]"
          : isGenerating
            ? "border-primary/40 shadow-[0_0_30px_rgba(0,242,255,0.08)]"
            : "border-white/10 hover:border-white/20 focus-within:border-primary/50 focus-within:shadow-[0_0_25px_rgba(0,242,255,0.05)]",
      )}
    >
      {/* Top Dock Bar: HeroUI Button Segmented Verbosity Selector & Mode Hint */}
      <div className="flex items-center justify-between px-3.5 py-2 border-b border-white/5 text-[10px] select-none">
        <div className="flex items-center gap-1 bg-black/40 p-0.5 rounded-xl border border-white/5">
          {(["concise", "balanced", "deep_dive"] as VerbosityMode[]).map((mode) => {
            const isSelected = verbosity === mode;
            return (
              <Button
                key={mode}
                size="sm"
                variant={isSelected ? "primary" : "ghost"}
                onPress={() => onVerbosityChange(mode)}
                className={cn(
                  "h-6 px-2.5 rounded-lg font-mono uppercase tracking-wider text-[9px] transition-all duration-200 cursor-pointer",
                  isSelected
                    ? "bg-white/15 text-primary font-bold shadow-sm border border-primary/30"
                    : "text-foreground/40 hover:text-foreground/80 hover:bg-white/[0.03]",
                )}
              >
                {VERBOSITY_CONFIG[mode].label}
              </Button>
            );
          })}
        </div>

        <span className="hidden sm:inline-block text-[9px] font-mono text-foreground/40 tracking-tight">
          {VERBOSITY_CONFIG[verbosity].description}
        </span>
      </div>

      {/* Input area using HeroUI TextArea */}
      <div className="flex items-end gap-3 p-3">
        <TextArea
          ref={textareaRef}
          rows={1}
          value={input}
          disabled={isLocked}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={
            isLocked
              ? "System locked awaiting parameter resolution..."
              : "Ask BudAI about spending trends, cash flow, stochastic forecast, or bucket transfers..."
          }
          className="flex-1 max-h-[180px] min-h-[40px] py-1 px-1 text-sm font-sans bg-transparent resize-none text-foreground placeholder:text-foreground/30 focus:outline-none custom-scrollbar leading-relaxed border-none shadow-none focus-within:ring-0 [&>div]:bg-transparent [&>div]:border-none"
        />

        <div className="flex items-center gap-1.5 shrink-0 mb-1">
          {isGenerating ? (
            <Button
              isIconOnly
              size="sm"
              onPress={stop}
              className="w-9 h-9 min-w-9 rounded-xl flex items-center justify-center bg-danger/20 text-danger hover:bg-danger/30 border border-danger/40 shadow-[0_0_15px_rgba(243,18,96,0.15)] transition-all cursor-pointer"
            >
              <Square size={13} fill="currentColor" />
            </Button>
          ) : (
            <Button
              isIconOnly
              size="sm"
              onPress={handleSend}
              isDisabled={!input.trim() || isLocked}
              variant={input.trim() && !isLocked ? "primary" : "ghost"}
              className={cn(
                "w-9 h-9 min-w-9 rounded-xl flex items-center justify-center transition-all cursor-pointer",
                input.trim() && !isLocked
                  ? "bg-primary text-black font-bold shadow-[0_0_20px_rgba(0,242,255,0.3)] hover:opacity-90 active:scale-95"
                  : "bg-white/5 text-foreground/20 border border-white/5 cursor-not-allowed",
              )}
            >
              <Send size={15} />
            </Button>
          )}
        </div>
      </div>

      {/* Sub-dock footer with shortcut indicators */}
      <div className="flex items-center justify-between px-3.5 pb-2 text-[9px] font-mono text-foreground/30 select-none">
        <span className="flex items-center gap-1">
          <Sparkles size={9} className="text-primary/70" />
          <span>Live financial tracking active</span>
        </span>
        <span className="hidden sm:flex items-center gap-1.5">
          <span>{isMac ? "⌘O" : "Ctrl+O"} thinking</span>
          <span>•</span>
          <span>Enter to send</span>
          <CornerDownLeft size={9} />
          <span>• Shift+Enter newline</span>
        </span>
      </div>
    </motion.div>
  );
};
