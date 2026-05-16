"use client";

import React, { useState } from "react";
import { motion } from "framer-motion";
import { Sparkles, ArrowRight, RotateCcw, Info } from "lucide-react";
import { Button, Card, Skeleton, ScrollShadow, Surface } from "@heroui/react";
import { cn } from "@/lib/utils";

interface WidgetFlipCardProps {
  children: React.ReactNode;
  insight?: string;
  isLoading?: boolean;
  onDiscuss?: () => void;
  className?: string;
}

export default function WidgetFlipCard({
  children,
  insight,
  isLoading,
  onDiscuss,
  className,
}: WidgetFlipCardProps) {
  const [isFlipped, setIsFlipped] = useState(false);

  const toggleFlip = () => setIsFlipped(!isFlipped);

  return (
    <div className={cn("relative w-full h-full perspective-1000", className)}>
      <motion.div
        className="relative w-full h-full transition-all duration-700 preserve-3d"
        animate={{ rotateY: isFlipped ? 180 : 0 }}
        transition={{ type: "spring", stiffness: 260, damping: 20 }}
      >
        <Surface
          variant="transparent"
          className="absolute inset-0 w-full h-full backface-hidden z-10 p-0 m-0 border-none bg-transparent"
        >
          {children}

          <Button
            onPress={toggleFlip}
            className="absolute top-5 right-14 h-8 px-3 rounded-lg flex items-center justify-center gap-1.5 bg-neon-cyan/10 text-neon-cyan border border-neon-cyan/20 hover:bg-neon-cyan/20 z-50 transition-all shadow-[0_0_10px_rgba(0,229,255,0.2)] hover:scale-105 cursor-pointer"
          >
            <span className="text-xs font-bold whitespace-nowrap">
              Ask AI Advisor
            </span>
            <Sparkles size={14} className="animate-pulse shrink-0" />
          </Button>
        </Surface>

        <Surface
          variant="transparent"
          className="absolute inset-0 w-full h-full backface-hidden rotate-y-180 z-0 p-0 m-0 border-none bg-transparent"
        >
          <Card className="w-full h-full bg-[#0d1516]/90 backdrop-blur-3xl rounded-3xl p-0 flex flex-col border border-neon-cyan/20 shadow-[0_0_40px_rgba(0,229,255,0.15)] overflow-hidden">
            <Card.Header className="flex items-center gap-4 p-8 pb-4 shrink-0">
              <div className="w-12 h-12 rounded-xl bg-neon-cyan/10 border border-neon-cyan/20 flex items-center justify-center text-neon-cyan shadow-[inset_0_0_15px_rgba(0,229,255,0.1)] shrink-0">
                <Sparkles size={22} />
              </div>
              <div className="flex flex-col justify-center">
                <Card.Title className="text-white font-bold text-lg leading-tight tracking-tight m-0">
                  AI Advisor
                </Card.Title>
                <Card.Description className="text-white/50 text-[10px] font-bold uppercase tracking-widest mt-0.5 m-0">
                  Contextual Intelligence
                </Card.Description>
              </div>
            </Card.Header>

            <Card.Content className="flex-1 px-8 py-0 overflow-hidden">
              <ScrollShadow className="h-full pr-2">
                {isLoading ? (
                  <div className="space-y-4 pt-4">
                    <Skeleton className="rounded-lg h-4 w-3/4 bg-white/5" />
                    <Skeleton className="rounded-lg h-4 w-full bg-white/5" />
                    <Skeleton className="rounded-lg h-4 w-2/3 bg-white/5" />
                  </div>
                ) : (
                  <div className="flex flex-col gap-3 mt-4">
                    <Info size={16} className="text-white/70" />
                    <p className="text-white/90 text-sm leading-relaxed font-medium italic">
                      {insight ||
                        "I'm currently analyzing your financial data to provide actionable advice. Check back in a moment."}
                    </p>
                  </div>
                )}
              </ScrollShadow>
            </Card.Content>

            <Card.Footer className="p-8 pt-6 mt-auto border-t border-white/5 flex items-center gap-4">
              <Button
                onPress={onDiscuss}
                className="flex-1 bg-white/5 hover:bg-white/10 text-white font-bold h-12 rounded-xl flex items-center justify-center gap-2 border border-white/10 hover:border-white/20 transition-all cursor-pointer shadow-none"
              >
                Discuss with Advisor
                <ArrowRight size={18} />
              </Button>
              <Button
                isIconOnly
                onPress={toggleFlip}
                variant="ghost"
                className="w-12 h-12 min-w-12 rounded-xl text-[#8B8E98] hover:text-white border border-white/10 hover:bg-white/20 flex items-center justify-center transition-all cursor-pointer bg-transparent"
              >
                <RotateCcw size={18} />
              </Button>
            </Card.Footer>
          </Card>
        </Surface>
      </motion.div>
    </div>
  );
}
