import React from "react";
import { Loader2 } from "lucide-react";

export default function ProtectedLoading() {
  return (
    <div className="flex h-screen w-full items-center justify-center bg-transparent">
      <div className="flex flex-col items-center gap-4 text-primary">
        <Loader2 className="w-10 h-10 animate-spin" />
        <p className="text-xs font-black uppercase tracking-widest text-primary/70">
          Loading Dashboard...
        </p>
      </div>
    </div>
  );
}
