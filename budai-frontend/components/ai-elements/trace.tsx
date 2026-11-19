"use client";

import React, { createContext, useContext, useState } from "react";
import { ChevronRight, Loader2, CheckCircle2, AlertCircle } from "lucide-react";
import { cn } from "@/lib/utils";

interface TraceContextValue {
  isCollapsible?: boolean;
  isOpen?: boolean;
  toggle?: () => void;
}

const TraceContext = createContext<TraceContextValue>({});

export interface TraceProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function Trace({ className, children, ...props }: TraceProps) {
  return (
    <div
      className={cn("flex flex-col gap-1 w-full my-1 text-xs select-none", className)}
      {...props}
    >
      {children}
    </div>
  );
}

export interface TraceCollapsibleProps extends React.HTMLAttributes<HTMLDivElement> {
  open?: boolean;
  defaultOpen?: boolean;
  onOpenChange?: (open: boolean) => void;
  children: React.ReactNode;
}

export function TraceCollapsible({
  open: controlledOpen,
  defaultOpen = false,
  onOpenChange,
  className,
  children,
  ...props
}: TraceCollapsibleProps) {
  const [uncontrolledOpen, setUncontrolledOpen] = useState(defaultOpen);
  const isOpen = controlledOpen !== undefined ? controlledOpen : uncontrolledOpen;

  const toggle = () => {
    const next = !isOpen;
    if (controlledOpen === undefined) {
      setUncontrolledOpen(next);
    }
    onOpenChange?.(next);
  };

  return (
    <TraceContext.Provider value={{ isCollapsible: true, isOpen, toggle }}>
      <div
        className={cn(
          "w-fit max-w-full my-0.5 transition-all duration-200",
          className
        )}
        {...props}
      >
        {children}
      </div>
    </TraceContext.Provider>
  );
}

export interface TraceCollapsibleTriggerProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  children: React.ReactNode;
}

export function TraceCollapsibleTrigger({
  className,
  children,
  ...props
}: TraceCollapsibleTriggerProps) {
  const { isOpen, toggle } = useContext(TraceContext);

  return (
    <button
      type="button"
      onClick={toggle}
      className={cn(
        "inline-flex items-center gap-1.5 py-1 px-2 rounded-lg text-left text-muted-foreground hover:text-foreground hover:bg-white/[0.04] transition-colors cursor-pointer group outline-none text-xs",
        className
      )}
      {...props}
    >
      <div className="flex items-center gap-1.5 min-w-0">{children}</div>
      <ChevronRight
        size={12}
        className={cn(
          "text-muted-foreground/60 transition-transform duration-200 shrink-0 group-hover:text-foreground",
          isOpen && "rotate-90 text-foreground"
        )}
      />
    </button>
  );
}

export interface TraceCollapsibleContentProps
  extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
  keepMounted?: boolean;
}

export function TraceCollapsibleContent({
  className,
  children,
  keepMounted = false,
  ...props
}: TraceCollapsibleContentProps) {
  const { isOpen } = useContext(TraceContext);

  if (!isOpen && !keepMounted) return null;

  return (
    <div
      className={cn(
        "mt-1.5 pl-4 ml-1.5 border-l border-white/10 text-muted-foreground/90 leading-relaxed text-xs",
        !isOpen && keepMounted && "hidden",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export type TraceStatus = "running" | "error" | "completed" | undefined;

export interface TraceLineProps extends React.HTMLAttributes<HTMLDivElement> {
  status?: TraceStatus;
  children: React.ReactNode;
}

export function TraceLine({ status, className, children, ...props }: TraceLineProps) {
  return (
    <div
      className={cn("inline-flex items-center gap-1.5 min-w-0 text-foreground text-xs", className)}
      {...props}
    >
      {status === "running" && (
        <Loader2 size={12} className="animate-spin text-primary shrink-0" />
      )}
      {status === "error" && (
        <AlertCircle size={12} className="text-danger shrink-0" />
      )}
      {status === "completed" && (
        <CheckCircle2 size={12} className="text-muted-foreground/80 group-hover:text-foreground shrink-0" />
      )}
      {children}
    </div>
  );
}

export interface TraceIconProps extends React.HTMLAttributes<HTMLSpanElement> {
  children: React.ReactNode;
}

export function TraceIcon({ className, children, ...props }: TraceIconProps) {
  return (
    <span
      className={cn(
        "inline-flex items-center justify-center shrink-0 text-muted-foreground [&>svg]:size-3.5",
        className
      )}
      {...props}
    >
      {children}
    </span>
  );
}

export interface TraceTitleProps extends React.HTMLAttributes<HTMLSpanElement> {
  children: React.ReactNode;
}

export function TraceTitle({ className, children, ...props }: TraceTitleProps) {
  return (
    <span
      className={cn("font-medium text-foreground truncate text-xs leading-none", className)}
      {...props}
    >
      {children}
    </span>
  );
}

export interface TraceDescriptionProps extends React.HTMLAttributes<HTMLSpanElement> {
  children: React.ReactNode;
}

export function TraceDescription({ className, children, ...props }: TraceDescriptionProps) {
  return (
    <span
      className={cn("text-[10px] font-mono text-muted-foreground shrink-0 ml-1.5", className)}
      {...props}
    >
      {children}
    </span>
  );
}
