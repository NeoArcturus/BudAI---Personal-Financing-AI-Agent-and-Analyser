"use client";

import React from "react";

export default function GlobalBackground() {
  return (
    <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden select-none bg-black">
      <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,127,255,0.05)_0%,transparent_70%)] z-10" />
      <div className="absolute inset-0 bg-grid-pattern opacity-[0.1] z-10" />
      <div className="absolute inset-0 bg-black/20 z-20" />
    </div>
  );
}
