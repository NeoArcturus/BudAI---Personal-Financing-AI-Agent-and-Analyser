"use client";

import React from "react";
import { Button } from "@heroui/react";
import { useRouter } from "next/navigation";
import { FileQuestion, Home } from "lucide-react";

export default function NotFound() {
  const router = useRouter();

  return (
    <div className="min-h-screen bg-[#08090D] flex flex-col items-center justify-center p-6 text-center relative overflow-hidden bg-grid-pattern">
      {/* Ambient Glow */}
      <div className="absolute top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-125 h-125 bg-cyan-500/10 blur-[120px] rounded-full pointer-events-none" />

      <div className="obsidian-glass p-12 rounded-[2rem] border border-white/10 flex flex-col items-center gap-8 max-w-lg relative z-10">
        <div className="w-20 h-20 rounded-2xl bg-cyan-400/10 border border-cyan-400/20 flex items-center justify-center text-cyan-400">
          <FileQuestion size={40} />
        </div>

        <div className="space-y-2">
          <h1 className="text-4xl font-bold text-white tracking-tight">404</h1>
          <h2 className="text-xl font-semibold text-white/80">
            Page Not Found
          </h2>
          <p className="text-white/40 leading-relaxed">
            The financial record you are looking for has been archived or does
            not exist. Check the URL or return to the dashboard.
          </p>
        </div>

        <Button
          onPress={() => router.push("/")}
          className="bg-cyan-400 text-black font-bold px-8 h-12 rounded-lg neon-glow-primary hover:bg-cyan-300 flex items-center gap-2"
        >
          <Home size={18} />
          Return Home
        </Button>
      </div>
    </div>
  );
}
