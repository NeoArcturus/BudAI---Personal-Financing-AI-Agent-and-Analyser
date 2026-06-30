"use client";

import React from "react";
import Image from "next/image";

export default function GlobalBackground() {
  return (
    <div className="fixed inset-0 z-0 pointer-events-none overflow-hidden select-none bg-black">
      <Image
        src="/bg 3.jpg"
        alt="Background"
        fill
        className="object-cover opacity-100"
        priority
      />
      <div className="absolute inset-0 bg-black/10 z-10" />
      <div className="absolute inset-0 bg-linear-to-b from-transparent via-transparent to-[#0c131d]/80 z-20" />
    </div>
  );
}
