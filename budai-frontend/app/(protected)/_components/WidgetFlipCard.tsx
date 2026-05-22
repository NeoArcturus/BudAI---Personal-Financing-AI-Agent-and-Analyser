// WidgetFlipCard.tsx
"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import {
  Sparkles,
  ArrowRight,
  RotateCcw,
  Info,
  BrainCircuit,
} from "lucide-react";
import { Button, Card, Skeleton, ScrollShadow, Surface } from "@heroui/react";
import { cn } from "@/lib/utils";

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
          {children}

          <Button
            onPress={toggleFlip}
            className="absolute top-8 right-16 h-8 px-4 rounded-lg flex items-center justify-center gap-2 bg-primary/5 text-primary border-[0.5px] border-primary/20 hover:border-primary/50 z-50 transition-all shadow-[0_0_15px_rgba(0,127,255,0.05)] hover:shadow-[0_0_20px_rgba(0,127,255,0.2)] cursor-pointer"
          >
            <span className="text-[9px] font-black uppercase tracking-[0.2em] whitespace-nowrap">
              Link Advisor
            </span>
            <Sparkles
              size={12}
              className={cn(
                "shrink-0",
                showSkeleton ? "animate-spin" : "animate-pulse",
              )}
            />
          </Button>
        </Surface>

        <Surface
          variant="transparent"
          className="absolute inset-0 w-full h-full backface-hidden rotate-y-180 z-0 p-0 m-0 border-none bg-transparent"
        >
          <Card className="w-full h-full liquid-glass rounded-xl p-0 flex flex-col border-none shadow-inner overflow-hidden">
            <Card.Header className="flex items-center gap-5 p-10 pb-6 shrink-0 border-b-[0.5px] border-white/5">
              <div className="w-12 h-12 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shadow-[0_0_20px_rgba(0,127,255,0.05)] shrink-0">
                <BrainCircuit size={24} />
              </div>
              <div className="flex flex-col justify-center">
                <h3 className="text-foreground font-black text-xl tracking-tighter uppercase italic m-0">
                  Neural Insights
                </h3>
                <p className="text-primary/50 text-[9px] font-black uppercase tracking-[0.3em] mt-1.5 m-0">
                  Real-time Logic Stream
                </p>
              </div>
            </Card.Header>

            <Card.Content className="flex-1 px-10 py-0 overflow-hidden min-h-0 relative">
              <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,127,255,0.01)_0%,transparent_70%)] pointer-events-none" />
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
                        Advisory Protocol
                      </span>
                    </div>
                    <p className="text-foreground/70 text-[15px] leading-relaxed font-medium tracking-tight">
                      {insight ||
                        "Processing institutional data streams. Analyzing spend patterns for logic optimization."}
                    </p>
                  </div>
                )}
              </ScrollShadow>
            </Card.Content>

            <Card.Footer className="p-10 pt-6 mt-auto border-t-[0.5px] border-white/5 flex items-center gap-5 bg-white/[0.01]">
              <Button
                onPress={onDiscuss}
                isDisabled={showSkeleton}
                className="flex-1 bg-primary text-primary-foreground font-black uppercase tracking-[0.2em] text-[11px] h-14 rounded-xl shadow-[0_0_20px_rgba(0,127,255,0.3)] hover:shadow-[0_0_30px_rgba(0,127,255,0.5)] transition-all cursor-pointer border-none flex items-center justify-center gap-3"
              >
                Sync Advisor
                <ArrowRight size={18} />
              </Button>
              <Button
                isIconOnly
                onPress={toggleFlip}
                className="w-14 h-14 min-w-14 rounded-xl text-foreground/30 hover:text-foreground hover:bg-white/5 flex items-center justify-center transition-all cursor-pointer bg-transparent border-[0.5px] border-white/10"
              >
                <RotateCcw size={20} />
              </Button>
            </Card.Footer>
          </Card>
        </Surface>
      </motion.div>
    </div>
  );
}
