"use client";

import React, { useEffect } from "react";
import { Button } from "@heroui/react";
import { AlertCircle, RefreshCcw } from "lucide-react";

export default function Error({
  error,
  reset,
}: {
  error: Error & { digest?: string };
  reset: () => void;
}) {
  useEffect(() => {
    console.error(error);
  }, [error]);

  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center justify-center p-6 text-center relative overflow-hidden">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-125 h-125 bg-destructive/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="obsidian-glass p-12 rounded-[2rem] border border-border flex flex-col items-center gap-8 max-w-lg relative z-10">
        <div className="w-20 h-20 rounded-3xl bg-destructive/10 border border-destructive/20 flex items-center justify-center text-destructive shadow-[0_0_20px_rgba(239,68,68,0.2)]">
          <AlertCircle size={40} />
        </div>

        <div className="space-y-3">
          <h1 className="text-3xl font-black text-foreground tracking-tighter uppercase italic">
            System Friction Detected
          </h1>
          <p className="text-muted-foreground font-medium leading-relaxed">
            An unexpected error occurred during financial orchestration. The
            multi-agent system has been alerted.
          </p>
          <div className="bg-secondary/50 border border-border p-4 rounded-2xl mt-4 backdrop-blur-sm">
            <code className="text-xs text-destructive font-mono break-all font-bold">
              {error.message || "INTERNAL_EXECUTION_FAULT"}
            </code>
          </div>
        </div>

        <Button
          onPress={() => reset()}
          className="bg-destructive text-white font-black px-10 h-14 rounded-2xl neon-glow-alert hover:bg-destructive/80 flex items-center gap-2 transition-all hover:scale-105 active:scale-95 cursor-pointer uppercase tracking-widest text-xs"
        >
          <RefreshCcw size={18} />
          Retry Orchestration
        </Button>
      </div>
    </div>
  );
}
