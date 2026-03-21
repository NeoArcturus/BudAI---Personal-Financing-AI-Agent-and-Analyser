"use client";

import React from "react";

interface StatCardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
}

export default function StatCard({ title, value, icon }: StatCardProps) {
  return (
    <div className="bg-[#161B22] border border-slate-800 rounded-2xl p-6 transition-all hover:border-[#00FFAA]/20">
      <div className="flex items-center gap-3 mb-3">
        {icon}
        <p className="text-slate-500 text-[10px] font-bold uppercase tracking-widest">
          {title}
        </p>
      </div>
      <h2 className="text-2xl font-mono font-bold text-white">{value}</h2>
    </div>
  );
}
