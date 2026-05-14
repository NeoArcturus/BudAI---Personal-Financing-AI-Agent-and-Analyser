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
    <div className="min-h-screen bg-[#08090D] flex flex-col items-center justify-center p-6 text-center relative overflow-hidden bg-grid-pattern">
      {/* Ambient Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-[500px] h-[500px] bg-pink-500/10 blur-[120px] rounded-full pointer-events-none" />
      
      <div className="obsidian-glass p-12 rounded-[2rem] border border-white/10 flex flex-col items-center gap-8 max-w-lg relative z-10">
        <div className="w-20 h-20 rounded-2xl bg-pink-500/10 border border-pink-500/20 flex items-center justify-center text-pink-500">
          <AlertCircle size={40} />
        </div>
        
        <div className="space-y-2">
          <h1 className="text-2xl font-bold text-white tracking-tight">System Friction Detected</h1>
          <p className="text-white/40 leading-relaxed">
            An unexpected error occurred during financial orchestration. 
            The multi-agent system has been alerted.
          </p>
          <div className="bg-black/20 p-3 rounded-lg border border-white/5 mt-4">
            <code className="text-xs text-pink-500/70 font-mono break-all">
              {error.message || "Unknown Error"}
            </code>
          </div>
        </div>
        
        <Button 
          onPress={() => reset()}
          className="bg-pink-500 text-white font-bold px-8 h-12 rounded-lg neon-glow-alert hover:bg-pink-400 flex items-center gap-2"
        >
          <RefreshCcw size={18} />
          Retry Orchestration
        </Button>
      </div>
    </div>
  );
}
