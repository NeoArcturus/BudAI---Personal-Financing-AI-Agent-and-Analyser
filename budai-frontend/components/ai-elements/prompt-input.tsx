"use client";

import React, { useState, useRef, useEffect } from "react";
import { ArrowUp, Square, ChevronDown, Check, Plus, Paperclip, X } from "lucide-react";
import { cn } from "@/lib/utils";
import { HoverCard, HoverCardTrigger, HoverCardContent } from "@/components/ui/hover-card";
import { Progress } from "@/components/ui/progress";
import { Button as HeroButton } from "@heroui/react";

export type EffortLevel = "low" | "medium" | "high";
export type DetailLevel = EffortLevel | "concise" | "balanced" | "deep_dive";

function normalizeEffort(level?: DetailLevel): EffortLevel {
  if (level === "concise") return "low";
  if (level === "deep_dive") return "high";
  if (level === "low" || level === "medium" || level === "high") return level;
  return "medium";
}

export interface PromptInputProps {
  onSend: (text: string, effort: EffortLevel, files?: File[]) => void;
  isGenerating?: boolean;
  onStop?: () => void;
  isLocked?: boolean;
  detailLevel?: DetailLevel;
  onDetailLevelChange?: (level: DetailLevel) => void;
  currentTokens?: number;
  maxTokens?: number;
  inputTokens?: number;
  outputTokens?: number;
  reasoningTokens?: number;
  modelId?: string;
  placeholder?: string;
  className?: string;
}

export function PromptInput({
  onSend,
  isGenerating = false,
  onStop,
  isLocked = false,
  detailLevel = "medium",
  onDetailLevelChange,
  currentTokens = 0,
  maxTokens = 20224,
  inputTokens = 0,
  outputTokens = 0,
  reasoningTokens = 0,
  modelId = "qwen3.5-9b",
  placeholder = "Ask a question about your spending, bills, or bank balance...",
  className,
}: PromptInputProps) {
  const [text, setText] = useState("");
  const [files, setFiles] = useState<File[]>([]);
  const [isEffortOpen, setIsEffortOpen] = useState(false);
  const textareaRef = useRef<HTMLTextAreaElement>(null);
  const dropdownRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  const activeEffort = normalizeEffort(detailLevel);

  // Close dropdown on outside click
  useEffect(() => {
    const handleClickOutside = (e: MouseEvent) => {
      if (dropdownRef.current && !dropdownRef.current.contains(e.target as Node)) {
        setIsEffortOpen(false);
      }
    };
    if (isEffortOpen) {
      document.addEventListener("mousedown", handleClickOutside);
    }
    return () => document.removeEventListener("mousedown", handleClickOutside);
  }, [isEffortOpen]);

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
    if ((!text.trim() && files.length === 0) || isGenerating || isLocked) return;
    onSend(text.trim(), activeEffort, files);
    setText("");
    setFiles([]);
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

  const canSend = (text.trim().length > 0 || files.length > 0) && !isGenerating && !isLocked;

  // Context calculations
  const safeTokens = Math.max(0, currentTokens);
  const safeLimit = maxTokens > 0 ? maxTokens : 20224;
  const percent = Math.min(100, Math.round((safeTokens / safeLimit) * 100));
  const isHighContext = percent >= 85;
  const isMediumContext = percent >= 65 && percent < 85;

  const formattedUsed =
    safeTokens >= 1000 ? `${(safeTokens / 1000).toFixed(1)}k` : `${safeTokens}`;
  const formattedLimit =
    safeLimit >= 1000 ? `${(safeLimit / 1000).toFixed(0)}k` : `${safeLimit}`;

  return (
    <div
      className={cn(
        "w-full rounded-2xl border transition-all duration-300 shadow-xl backdrop-blur-2xl bg-[#121d2c]/85 border-[#1e2d40] hover:border-[#273a52] focus-within:border-primary/50 focus-within:shadow-[0_0_25px_rgba(0,242,255,0.05)] flex flex-col p-2.5 gap-1.5",
        isLocked && "opacity-60 pointer-events-none",
        isGenerating && "border-primary/40 shadow-[0_0_30px_rgba(0,242,255,0.08)]",
        className
      )}
    >
      {/* Hidden file input */}
      <input
        ref={fileInputRef}
        type="file"
        multiple
        className="hidden"
        onChange={(e) => {
          if (e.target.files) {
            setFiles((prev) => [...prev, ...Array.from(e.target.files!)]);
          }
        }}
      />

      {/* Attachment Chips */}
      {files.length > 0 && (
        <div className="flex flex-wrap gap-1.5 px-1 pt-1">
          {files.map((file, idx) => (
            <div
              key={idx}
              className="inline-flex items-center gap-1.5 px-2.5 py-1 rounded-full bg-[#1b2738] text-foreground text-[11px] border border-white/10"
            >
              <Paperclip size={11} className="text-primary shrink-0" />
              <span className="truncate max-w-[150px]">{file.name}</span>
              <button
                type="button"
                onClick={() => setFiles((prev) => prev.filter((_, i) => i !== idx))}
                className="text-muted-foreground hover:text-foreground cursor-pointer ml-0.5"
              >
                <X size={11} />
              </button>
            </div>
          ))}
        </div>
      )}

      {/* Main Textarea */}
      <textarea
        ref={textareaRef}
        rows={1}
        value={text}
        onChange={(e) => setText(e.target.value)}
        onKeyDown={handleKeyDown}
        disabled={isLocked}
        placeholder={isLocked ? "Approval required to proceed..." : placeholder}
        className="w-full bg-transparent text-foreground placeholder:text-muted-foreground/50 text-xs px-2 py-1.5 outline-none resize-none min-h-[36px] max-h-[180px] scrollbar-hide font-sans"
      />

      {/* Action Toolbar */}
      <div className="flex items-center justify-between gap-2 pt-2 border-t border-[#1a2636] px-1 select-none">
        
        {/* Left Controls: Attach + Effort + Model Context */}
        <div className="flex items-center gap-1.5 flex-wrap">
          
          {/* Attach Button */}
          <HeroButton
            isIconOnly
            size="sm"
            variant="ghost"
            onPress={() => fileInputRef.current?.click()}
            aria-label="Attach documents"
            className="w-7 h-7 min-w-7 rounded-full text-muted-foreground hover:text-foreground hover:bg-white/[0.06] transition-colors cursor-pointer"
          >
            <Plus size={15} />
          </HeroButton>

          {/* Effort Selector */}
          <div ref={dropdownRef} className="relative">
            <button
              type="button"
              onClick={() => setIsEffortOpen((prev) => !prev)}
              className={cn(
                "flex items-center gap-1 px-2.5 py-1 rounded-lg border text-[11px] font-medium transition-colors cursor-pointer outline-none",
                isEffortOpen
                  ? "bg-white/10 border-primary/40 text-foreground"
                  : "bg-white/[0.03] hover:bg-white/[0.08] border-[#1e2d40] text-muted-foreground hover:text-foreground"
              )}
            >
              <span>
                Effort: <strong className="text-foreground font-semibold capitalize">{activeEffort}</strong>
              </span>
              <ChevronDown
                size={11}
                className={cn(
                  "text-muted-foreground transition-transform duration-200",
                  isEffortOpen && "rotate-180 text-foreground"
                )}
              />
            </button>

            {isEffortOpen && (
              <div className="absolute bottom-full mb-2 left-0 w-36 rounded-xl bg-[#0e1622]/95 border border-[#1e2d40] backdrop-blur-2xl p-1 shadow-2xl z-50 flex flex-col gap-0.5 text-foreground animate-in fade-in zoom-in-95 duration-100">
                <div className="px-2 py-1 text-[10px] font-semibold text-muted-foreground/60 uppercase tracking-wider">
                  Effort
                </div>
                {(["low", "medium", "high"] as EffortLevel[]).map((level) => {
                  const isSelected = activeEffort === level;
                  return (
                    <button
                      key={level}
                      type="button"
                      onClick={() => {
                        onDetailLevelChange?.(level);
                        setIsEffortOpen(false);
                      }}
                      className={cn(
                        "w-full text-left px-2.5 py-1.5 rounded-lg text-xs transition-colors cursor-pointer flex items-center justify-between",
                        isSelected
                          ? "bg-primary/10 text-primary font-semibold"
                          : "hover:bg-white/5 text-foreground/85 hover:text-foreground"
                      )}
                    >
                      <span className="capitalize">{level}</span>
                      {isSelected && <Check size={12} className="text-primary" />}
                    </button>
                  );
                })}
              </div>
            )}
          </div>

          {/* Model Context Popover */}
          <HoverCard>
            <HoverCardTrigger>
              <button
                type="button"
                className="h-7 px-2.5 rounded-lg bg-white/[0.03] hover:bg-white/[0.08] border border-[#1e2d40] text-muted-foreground hover:text-foreground text-[11px] font-mono flex items-center gap-1.5 cursor-pointer transition-colors"
              >
                <span>
                  Context: <strong className="text-foreground font-semibold">{formattedUsed}/{formattedLimit}</strong>
                </span>
                <span
                  className={cn(
                    "font-bold",
                    isHighContext
                      ? "text-danger"
                      : isMediumContext
                      ? "text-amber-400"
                      : "text-primary"
                  )}
                >
                  ({percent}%)
                </span>
              </button>
            </HoverCardTrigger>
            <HoverCardContent className="w-64 p-3.5 rounded-2xl bg-[#0e1622]/95 border border-[#1e2d40] backdrop-blur-2xl text-xs text-foreground shadow-2xl flex flex-col gap-2.5">
              <div className="flex justify-between items-center text-xs font-semibold">
                <span>Model Context</span>
                <span className="text-primary font-mono font-bold">{percent}%</span>
              </div>
              <Progress value={percent} className="h-1.5 bg-white/10" />
              <div className="flex flex-col gap-1 text-[11px] text-muted-foreground border-t border-white/5 pt-2">
                <div className="flex justify-between">
                  <span>Input Tokens</span>
                  <span className="font-mono text-foreground font-medium">{inputTokens.toLocaleString()}</span>
                </div>
                <div className="flex justify-between">
                  <span>Output Tokens</span>
                  <span className="font-mono text-foreground font-medium">{outputTokens.toLocaleString()}</span>
                </div>
                {reasoningTokens > 0 && (
                  <div className="flex justify-between text-[10px] text-muted-foreground/75">
                    <span>Reasoning Tokens</span>
                    <span className="font-mono text-foreground/80">{reasoningTokens.toLocaleString()}</span>
                  </div>
                )}
              </div>
              <div className="flex justify-between text-[10px] text-muted-foreground/60 border-t border-white/5 pt-1.5">
                <span>Total: {safeTokens.toLocaleString()}</span>
                <span>Max: {safeLimit.toLocaleString()}</span>
              </div>
            </HoverCardContent>
          </HoverCard>

        </div>

        {/* Right Controls: Send / Stop */}
        {isGenerating ? (
          <HeroButton
            isIconOnly
            size="sm"
            onPress={onStop}
            aria-label="Stop generation"
            className="w-7 h-7 min-w-7 rounded-full bg-danger/20 hover:bg-danger/30 text-danger border border-danger/40 transition-all cursor-pointer"
          >
            <Square size={11} className="fill-current" />
          </HeroButton>
        ) : (
          <HeroButton
            isIconOnly
            size="sm"
            onPress={handleSend}
            isDisabled={!canSend}
            aria-label="Send message"
            className={cn(
              "w-7 h-7 min-w-7 rounded-full transition-all cursor-pointer shadow-xs",
              canSend
                ? "bg-primary text-primary-foreground hover:opacity-90 shadow-[0_0_15px_rgba(0,242,255,0.4)]"
                : "bg-white/[0.06] text-muted-foreground/40 cursor-not-allowed"
            )}
          >
            <ArrowUp size={13} strokeWidth={2.5} />
          </HeroButton>
        )}

      </div>
    </div>
  );
}
