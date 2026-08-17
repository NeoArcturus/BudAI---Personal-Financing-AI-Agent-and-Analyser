"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Sparkles,
  ArrowRight,
  RotateCcw,
  MessageSquare,
  Info,
} from "lucide-react";
import { Button, Card, Skeleton, ScrollShadow, Surface } from "@heroui/react";
import { cn } from "@/lib/utils";

export const FlipCardContext = React.createContext({
  isFlipped: false,
  toggleFlip: () => {},
});

export const useFlipCard = () => React.useContext(FlipCardContext);

export const FlipButton = ({ className }: { className?: string }) => {
  const { toggleFlip } = useFlipCard();
  return (
    <Button
      isIconOnly
      size="sm"
      variant="ghost"
      onPress={toggleFlip}
      className={cn("text-foreground/50 hover:text-foreground transition-all rounded-full bg-transparent border-none", className)}
    >
      <RotateCcw size={16} />
    </Button>
  );
};

interface WidgetFlipCardProps {
  children: React.ReactNode;
  insight?: string;
  isLoading?: boolean;
  isDataLoading?: boolean;
  onDiscuss?: () => void;
  className?: string;
}

export default function WidgetFlipCard({
  children,
  insight,
  isLoading,
  isDataLoading,
  onDiscuss,
  className,
}: WidgetFlipCardProps) {
  const [isFlipped, setIsFlipped] = useState(false);

  const toggleFlip = () => setIsFlipped(!isFlipped);

  const showSkeleton = isLoading || isDataLoading;

  return (
    <div className={cn("relative w-full h-full perspective-1000", className)}>
      <motion.div
        className="relative w-full h-full transition-all duration-500 preserve-3d"
        animate={{ rotateY: isFlipped ? 180 : 0 }}
        transition={{ type: "spring", stiffness: 350, damping: 25 }}
      >
        <Surface
          variant="transparent"
          className="absolute inset-0 w-full h-full backface-hidden z-10 p-0 m-0 border-none bg-transparent"
        >
          <FlipCardContext.Provider value={{ isFlipped, toggleFlip }}>
            {children}
          </FlipCardContext.Provider>
        </Surface>

        <Surface
          variant="transparent"
          className="absolute inset-0 w-full h-full backface-hidden rotate-y-180 z-0 p-0 m-0 border-none bg-transparent"
        >
          <Card className="w-full h-full liquid-glass rounded-xl p-0 flex flex-col border-none shadow-inner overflow-hidden">
            <Card.Header className="flex items-center gap-5 p-10 pb-6 shrink-0 border-b-[0.5px] border-white/5">
              <div className="w-12 h-12 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shadow-[0_0_20px_rgba(0,242,255,0.05)] shrink-0">
                <MessageSquare size={24} />
              </div>
              <div className="flex flex-col justify-center">
                <h3 className="text-foreground font-black text-[10px] tracking-widest uppercase italic m-0">
                  Analysis
                </h3>
                <p className="text-primary/50 text-[8px] font-mono tracking-widest mt-1 m-0">
                  System Context
                </p>
              </div>
            </Card.Header>

            <Card.Content className="flex-1 px-10 py-0 overflow-hidden min-h-0 relative">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,242,255,0.01)_0%,transparent_70%)] pointer-events-none" />
              <ScrollShadow
                hideScrollBar
                className="h-full pr-2 overflow-y-auto scrollbar-hide [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] relative z-10"
              >
                {showSkeleton ? (
                  <div className="space-y-6 pt-10">
                    <Skeleton
                      className="rounded-lg h-2 w-3/4 bg-white/5"
                      animationType="shimmer"
                    />
                    <Skeleton
                      className="rounded-lg h-2 w-full bg-white/5"
                      animationType="shimmer"
                    />
                    <Skeleton
                      className="rounded-lg h-2 w-2/3 bg-white/5"
                      animationType="shimmer"
                    />
                  </div>
                ) : (
                  <div className="flex flex-col gap-6 mt-10">
                    <div className="flex items-center gap-3 text-primary/30">
                      <div className="w-1.5 h-1.5 rounded-full bg-primary/40 animate-pulse" />
                      <span className="text-[9px] font-black uppercase tracking-[0.4em]">
                        Context
                      </span>
                    </div>
                    <p className="text-foreground/70 text-[15px] leading-relaxed font-medium tracking-tight">
                      {insight ||
                        "Analyzing your financial data. Identifying trends and opportunities for optimization."}
                    </p>
                  </div>
                )}
              </ScrollShadow>
            </Card.Content>

            <Card.Footer className="p-4 mt-auto border-t-[0.5px] border-white/5 flex items-center justify-end bg-white/[0.01]">
              <Button
                isIconOnly
                size="sm"
                variant="ghost"
                onPress={toggleFlip}
                className="w-8 h-8 min-w-8 text-foreground/20 hover:text-foreground hover:bg-white/5 transition-all rounded-md cursor-pointer bg-transparent border-none"
              >
                <RotateCcw size={16} />
              </Button>
            </Card.Footer>
          </Card>
        </Surface>
      </motion.div>
    </div>
  );
}
