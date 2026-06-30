"use client";

import React from "react";
import { Button } from "@heroui/react";
import { useRouter } from "next/navigation";
import { FileQuestion, Home } from "lucide-react";

export default function NotFound() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-transparent flex flex-col items-center justify-center p-6 text-center relative overflow-hidden">
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-125 h-125 bg-primary/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="obsidian-glass p-12 rounded-[2rem] border border-border flex flex-col items-center gap-8 max-w-lg relative z-10">
        <div className="w-20 h-20 rounded-3xl bg-primary/10 border border-primary/20 flex items-center justify-center text-primary shadow-[0_0_20px_rgba(0,242,255,0.2)]">
          <FileQuestion size={40} />
        </div>

        <div className="space-y-3">
          <h1 className="text-6xl font-black text-foreground tracking-tighter italic">404</h1>
          <h2 className="text-xl font-bold text-foreground/80 uppercase tracking-widest">
            Record Not Found
          </h2>
          <p className="text-muted-foreground font-medium leading-relaxed">
            The financial record you are looking for has been archived or does
            not exist in the current ledger.
          </p>
        </div>

        <Button
          onPress={() => router.push("/")}
          className="bg-linear-to-r from-[#7000ff] to-[#00f2ff] text-white border-none font-black px-10 h-14 rounded-2xl neon-glow-primary hover:bg-primary/80 flex items-center gap-2 transition-all hover:scale-105 active:scale-95 cursor-pointer uppercase tracking-widest text-xs"
        >
          <Home size={18} />
          Return to Hub
        </Button>
      </div>
    </div>
  );
}
