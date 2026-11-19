"use client";

import React, { useState } from "react";
import { Copy, Check, RefreshCcw } from "lucide-react";
import { cn } from "@/lib/utils";

export interface MessageProps extends React.HTMLAttributes<HTMLDivElement> {
  align?: "start" | "end";
  children: React.ReactNode;
}

export function Message({ align = "start", className, children, ...props }: MessageProps) {
  return (
    <div
      className={cn(
        "w-full flex transition-all duration-200",
        align === "end" ? "justify-end" : "justify-start",
        className
      )}
      {...props}
    >
      <div
        className={cn(
          "flex gap-3 max-w-3xl w-full",
          align === "end" ? "flex-row-reverse items-start" : "flex-row items-start"
        )}
      >
        {children}
      </div>
    </div>
  );
}

export interface MessageContentProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function MessageContent({ className, children, ...props }: MessageContentProps) {
  return (
    <div className={cn("flex flex-col min-w-0 flex-1 gap-2", className)} {...props}>
      {children}
    </div>
  );
}

export interface BubbleProps extends React.HTMLAttributes<HTMLDivElement> {
  align?: "start" | "end";
  variant?: "default" | "muted";
  children: React.ReactNode;
}

export function Bubble({
  align = "end",
  variant = "muted",
  className,
  children,
  ...props
}: BubbleProps) {
  return (
    <div
      className={cn(
        "rounded-2xl px-4 py-2.5 text-xs text-foreground leading-relaxed transition-all shadow-xs",
        variant === "muted"
          ? "bg-foreground/5 border border-white/10 backdrop-blur-md"
          : "bg-primary/10 border border-primary/20",
        align === "end" ? "ml-auto max-w-xl" : "mr-auto max-w-xl",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export interface BubbleContentProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function BubbleContent({ className, children, ...props }: BubbleContentProps) {
  return (
    <div className={cn("text-xs leading-relaxed whitespace-pre-wrap break-words", className)} {...props}>
      {children}
    </div>
  );
}

export interface MessageFooterProps extends React.HTMLAttributes<HTMLDivElement> {
  textToCopy?: string;
  onRetry?: () => void;
  children?: React.ReactNode;
}

export function MessageFooter({
  textToCopy,
  onRetry,
  className,
  children,
  ...props
}: MessageFooterProps) {
  const [copied, setCopied] = useState(false);

  const handleCopy = async () => {
    if (!textToCopy) return;
    try {
      await navigator.clipboard.writeText(textToCopy);
      setCopied(true);
      setTimeout(() => setCopied(false), 2000);
    } catch {
      // Clipboard write failed
    }
  };

  return (
    <div className={cn("flex items-center gap-1.5 mt-1 select-none", className)} {...props}>
      {textToCopy && (
        <button
          type="button"
          onClick={handleCopy}
          aria-label="Copy message"
          className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] text-muted-foreground hover:text-foreground hover:bg-foreground/5 transition-colors cursor-pointer border border-transparent hover:border-white/10"
        >
          {copied ? (
            <>
              <Check size={11} className="text-emerald-400" />
              <span className="text-emerald-400 font-medium">Copied</span>
            </>
          ) : (
            <>
              <Copy size={11} />
              <span>Copy</span>
            </>
          )}
        </button>
      )}

      {onRetry && (
        <button
          type="button"
          onClick={onRetry}
          aria-label="Retry message"
          className="flex items-center gap-1 px-2 py-1 rounded-md text-[10px] text-muted-foreground hover:text-foreground hover:bg-foreground/5 transition-colors cursor-pointer border border-transparent hover:border-white/10"
        >
          <RefreshCcw size={11} />
          <span>Try again</span>
        </button>
      )}

      {children}
    </div>
  );
}
