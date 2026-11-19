"use client";

import React, { useState, useRef, useEffect } from "react";
import { ArrowUp, Square, ChevronDown, Check } from "lucide-react";
import { cn } from "@/lib/utils";

export type DetailLevel = "concise" | "balanced" | "deep_dive";

const DETAIL_OPTIONS: Record<DetailLevel, { label: string; hint: string }> = {
  concise: { label: "Brief", hint: "Direct numbers and short answers" },
  balanced: { label: "Normal", hint: "Standard financial summary" },
  deep_dive: { label: "Detailed", hint: "Full breakdown with itemized data" },
};

export interface PromptInputProps {
  onSend: (text: string, detail: DetailLevel) => void;
  isGenerating?: boolean;
  onStop?: () => void;
  isLocked?: boolean;
  detailLevel?: DetailLevel;
  onDetailLevelChange?: (level: DetailLevel) => void;
  placeholder?: string;
  className?: string;
}

export function PromptInput({
  onSend,
  isGenerating = false,
  onStop,
  isLocked = false,
  detailLevel = "balanced",
  onDetailLevelChange,
  placeholder = "Ask your financial advisor a question...",
  className,
}: PromptInputProps) {
  const [text, setText] = useState("");
  const [isDetailOpen, setIsDetailOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsDetailOpen(false);
      }
    };
    if (isDetailOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isDetailOpen]);

  // Auto-resize textarea height
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = "auto";
      textareaRef.current.style.height = `${Math.min(
        textareaRef.current.scrollHeight,
        180
      )}px`;
    }
  }, [text]);

  const handleSend = () => {
    if (!text.trim() || isGenerating || isLocked) return;
    onSend(text.trim(), detailLevel);
    setText("");
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

  const canSend = text.trim().length > 0 && !isGenerating && !isLocked;

  return (
    <div
      className={cn(
        "w-full rounded-2xl border transition-all duration-300 shadow-xl backdrop-blur-2xl bg-[#0d0e14]/80 flex flex-col p-2.5",
        isLocked
          ? "border-amber-500/30 opacity-60 pointer-events-none"
          : isGenerating
          ? "border-primary/40 shadow-[0_0_30px_rgba(0,242,255,0.08)]"
          : "border-white/10 hover:border-white/20 focus-within:border-primary/50 focus-within:shadow-[0_0_25px_rgba(0,242,255,0.05)]",
        className
      )}
    >
      <textarea
        ref={textareaRef}
        rows={1}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isLocked}
        placeholder={isLocked ? "Approval required to proceed..." : placeholder}
        className="w-full bg-transparent text-foreground placeholder:text-muted-foreground/60 text-xs px-2 py-1.5 outline-none resize-none min-h-[36px] max-h-[180px] scrollbar-hide"
      />

      <div className="flex items-center justify-between gap-2 pt-2 border-t border-white/5 px-1 select-none">
        {/* Detail Level Selector */}
        <div ref={dropdownRef} className="relative">
          <button
            type="button"
            onClick={() => setIsDetailOpen((prev) => !prev)}
            className={cn(
              "flex items-center gap-1.5 px-2.5 py-1 rounded-lg border text-[11px] font-medium transition-colors cursor-pointer outline-none",
              isDetailOpen
                ? "bg-white/10 border-primary/40 text-foreground"
                : "bg-white/[0.03] hover:bg-white/[0.08] border-white/10 text-muted-foreground hover:text-foreground"
            )}
          >
            <span>
              Detail: <strong className="text-foreground font-semibold">{DETAIL_OPTIONS[detailLevel]?.label || "Normal"}</strong>
            </span>
            <ChevronDown
              size={11}
              className={cn(
                "text-muted-foreground transition-transform duration-200",
                isDetailOpen && "rotate-180 text-foreground"
              )}
            />
          </button>

          {isDetailOpen && (
            <div className="absolute bottom-full mb-2 left-0 w-60 rounded-2xl bg-[#0d0e14]/95 border border-white/15 backdrop-blur-2xl p-1.5 shadow-2xl z-50 flex flex-col gap-1 text-foreground animate-in fade-in zoom-in-95 duration-100">
              <div className="px-2.5 py-1 text-[10px] font-semibold text-muted-foreground/70 uppercase tracking-wider">
                Response Detail
              </div>
              {(Object.keys(DETAIL_OPTIONS) as DetailLevel[]).map((level) => {
                const isSelected = detailLevel === level;
                return (
                  <button
                    key={level}
                    type="button"
                    onClick={() => {
                      onDetailLevelChange?.(level);
                      setIsDetailOpen(false);
                    }}
                    className={cn(
                      "w-full text-left px-3 py-2 rounded-xl text-xs transition-colors cursor-pointer flex flex-col items-start gap-0.5",
                      isSelected
                        ? "bg-primary/10 text-primary border border-primary/20"
                        : "hover:bg-white/5 text-foreground/80 hover:text-foreground border border-transparent"
                    )}
                  >
                    <div className="flex items-center justify-between w-full">
                      <span className="font-semibold text-xs text-foreground">
                        {DETAIL_OPTIONS[level].label}
                      </span>
                      {isSelected && <Check size={13} className="text-primary" />}
                    </div>
                    <span className="text-[10px] text-muted-foreground">
                      {DETAIL_OPTIONS[level].hint}
                    </span>
                  </button>
                );
              })}
            </div>
          )}
        </div>

        {/* Action Button: Stop if generating, Send if idle */}
        {isGenerating ? (
          <button
            type="button"
            onClick={onStop}
            aria-label="Stop generation"
            className="size-7 rounded-full bg-danger/20 hover:bg-danger/30 text-danger border border-danger/40 flex items-center justify-center transition-all cursor-pointer"
          >
            <Square size={11} className="fill-current" />
          </button>
        ) : (
          <button
            type="button"
            onClick={handleSend}
            disabled={!canSend}
            aria-label="Send message"
            className={cn(
              "size-7 rounded-full flex items-center justify-center transition-all cursor-pointer shadow-xs",
              canSend
                ? "bg-primary text-primary-foreground hover:opacity-90 shadow-[0_0_15px_rgba(0,242,255,0.4)]"
                : "bg-foreground/10 text-muted-foreground cursor-not-allowed opacity-50"
            )}
          >
            <ArrowUp size={13} strokeWidth={2.5} />
          </button>
        )}
      </div>
    </div>
  );
}
