"use client";

import React, {
  createContext,
  useContext,
  useRef,
  useState,
  useEffect,
  useCallback,
} from "react";
import { ArrowDown } from "lucide-react";
import { cn } from "@/lib/utils";

interface MessageScrollerContextValue {
  autoScroll: boolean;
  isAtBottom: boolean;
  scrollToBottom: (smooth?: boolean) => void;
  viewportRef: React.RefObject<HTMLDivElement | null>;
  setViewportRef: (el: HTMLDivElement | null) => void;
  setIsAtBottom: (atBottom: boolean) => void;
}

const MessageScrollerContext = createContext<MessageScrollerContextValue | null>(null);

export function useMessageScroller() {
  const context = useContext(MessageScrollerContext);
  if (!context) {
    throw new Error("useMessageScroller must be used within a MessageScrollerProvider");
  }
  return context;
}

export interface MessageScrollerProviderProps {
  autoScroll?: boolean;
  children: React.ReactNode;
}

export function MessageScrollerProvider({
  autoScroll = true,
  children,
}: MessageScrollerProviderProps) {
  const viewportRef = useRef<HTMLDivElement | null>(null);
  const [isAtBottom, setIsAtBottom] = useState(true);

  const scrollToBottom = useCallback((smooth = true) => {
    if (viewportRef.current) {
      viewportRef.current.scrollTo({
        top: viewportRef.current.scrollHeight,
        behavior: smooth ? "smooth" : "instant",
      });
      setIsAtBottom(true);
    }
  }, []);

  const setViewportRef = useCallback((el: HTMLDivElement | null) => {
    viewportRef.current = el;
  }, []);

  return (
    <MessageScrollerContext.Provider
      value={{
        autoScroll,
        isAtBottom,
        scrollToBottom,
        viewportRef,
        setViewportRef,
        setIsAtBottom,
      }}
    >
      {children}
    </MessageScrollerContext.Provider>
  );
}

export interface MessageScrollerProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function MessageScroller({ className, children, ...props }: MessageScrollerProps) {
  return (
    <div className={cn("relative flex-1 flex flex-col min-h-0 w-full overflow-hidden", className)} {...props}>
      {children}
    </div>
  );
}

export interface MessageScrollerViewportProps
  extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function MessageScrollerViewport({
  className,
  children,
  ...props
}: MessageScrollerViewportProps) {
  const { setViewportRef, autoScroll, isAtBottom, setIsAtBottom } = useMessageScroller();
  const localRef = useRef<HTMLDivElement | null>(null);

  const handleScroll = useCallback(() => {
    if (!localRef.current) return;
    const { scrollTop, scrollHeight, clientHeight } = localRef.current;
    const distanceToBottom = scrollHeight - scrollTop - clientHeight;
    // Within 64px is considered at bottom
    const atBottom = distanceToBottom <= 64;
    setIsAtBottom(atBottom);
  }, [setIsAtBottom]);

  // When DOM changes (stream appends tokens), auto-scroll only if already at bottom
  useEffect(() => {
    const el = localRef.current;
    if (!el || !autoScroll) return;

    const observer = new MutationObserver(() => {
      if (isAtBottom) {
        el.scrollTo({ top: el.scrollHeight, behavior: "instant" });
      }
    });

    observer.observe(el, { childList: true, subtree: true, characterData: true });
    return () => observer.disconnect();
  }, [autoScroll, isAtBottom]);

  return (
    <div
      ref={(el) => {
        localRef.current = el;
        setViewportRef(el);
      }}
      onScroll={handleScroll}
      className={cn(
        "flex-1 overflow-y-auto px-4 sm:px-8 py-6 relative z-10 scrollbar-hide",
        className
      )}
      {...props}
    >
      {children}
    </div>
  );
}

export interface MessageScrollerContentProps extends React.HTMLAttributes<HTMLDivElement> {
  children: React.ReactNode;
}

export function MessageScrollerContent({
  className,
  children,
  ...props
}: MessageScrollerContentProps) {
  return (
    <div className={cn("max-w-4xl mx-auto w-full flex flex-col gap-6", className)} {...props}>
      {children}
    </div>
  );
}

export interface MessageScrollerItemProps extends React.HTMLAttributes<HTMLDivElement> {
  messageId?: string;
  scrollAnchor?: boolean;
  children: React.ReactNode;
}

export function MessageScrollerItem({
  messageId,
  scrollAnchor = false,
  className,
  children,
  ...props
}: MessageScrollerItemProps) {
  const itemRef = useRef<HTMLDivElement | null>(null);

  // If this item is designated as a scroll anchor (e.g. user prompt), bring into view gracefully once mounted
  useEffect(() => {
    if (scrollAnchor && itemRef.current) {
      itemRef.current.scrollIntoView({ behavior: "smooth", block: "start" });
    }
  }, [scrollAnchor]);

  return (
    <div
      ref={itemRef}
      data-message-id={messageId}
      className={cn("w-full transition-all duration-200", className)}
      {...props}
    >
      {children}
    </div>
  );
}

export interface MessageScrollerButtonProps
  extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  label?: string;
}

export function MessageScrollerButton({
  label = "Scroll to live edge",
  className,
  ...props
}: MessageScrollerButtonProps) {
  const { isAtBottom, scrollToBottom } = useMessageScroller();

  if (isAtBottom) return null;

  return (
    <button
      type="button"
      onClick={() => scrollToBottom(true)}
      aria-label={label}
      className={cn(
        "absolute bottom-4 right-8 z-30 flex items-center gap-1.5 px-3 py-1.5 rounded-full bg-[#0b0c14]/90 backdrop-blur-xl border border-white/20 text-foreground text-xs font-semibold shadow-2xl hover:bg-white/10 transition-all cursor-pointer",
        className
      )}
      {...props}
    >
      <ArrowDown size={12} className="text-primary animate-bounce" />
      <span className="text-[11px] font-medium">{label}</span>
    </button>
  );
}
