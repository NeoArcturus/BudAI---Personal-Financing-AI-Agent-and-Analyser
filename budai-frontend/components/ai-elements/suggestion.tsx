"use client";

import React from "react";
import { Button } from "@heroui/react";
import { cn } from "@/lib/utils";

export interface SuggestionsProps extends Omit<React.HTMLAttributes<HTMLDivElement>, "onSelect"> {
  suggestions?: string[];
  onSelect?: (suggestion: string) => void;
  children?: React.ReactNode;
}

export function Suggestions({
  suggestions,
  onSelect,
  className,
  children,
  ...props
}: SuggestionsProps) {
  if (!children && (!suggestions || suggestions.length === 0)) return null;

  return (
    <div
      className={cn(
        "flex items-center gap-2 overflow-x-auto scrollbar-hide py-1.5 px-0.5 overscroll-x-contain select-none",
        className
      )}
      {...props}
    >
      {children ||
        suggestions?.map((suggestion) => (
          <Suggestion
            key={suggestion}
            suggestion={suggestion}
            onClick={() => onSelect?.(suggestion)}
          />
        ))}
    </div>
  );
}

export interface SuggestionProps {
  suggestion: string;
  onClick?: () => void;
  className?: string;
}

export function Suggestion({ suggestion, onClick, className }: SuggestionProps) {
  return (
    <Button
      size="sm"
      variant="secondary"
      onPress={onClick}
      className={cn(
        "shrink-0 text-[11px] font-normal h-7 px-3 rounded-full bg-[#121d2b]/80 hover:bg-[#182638] text-muted-foreground hover:text-foreground border border-[#1f2d40] hover:border-primary/40 transition-all cursor-pointer whitespace-nowrap shadow-xs",
        className
      )}
    >
      {suggestion}
    </Button>
  );
}
