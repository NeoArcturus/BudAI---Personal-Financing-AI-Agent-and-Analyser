"use client";

import React from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";

interface StatCardProps {
  title: string;
  value: string;
  icon: React.ReactNode;
}

export default function StatCard({ title, value, icon }: StatCardProps) {
  return (
    <Card className="bg-[#161B22] border-slate-800 transition-all hover:border-[#00FFAA]/20">
      <CardHeader className="flex flex-row items-center space-y-0 pb-2 gap-3">
        {icon}
        <CardTitle className="text-slate-500 text-[10px] font-bold uppercase tracking-widest">
          {title}
        </CardTitle>
      </CardHeader>
      <CardContent>
        <div className="text-2xl font-mono font-bold text-white">{value}</div>
      </CardContent>
    </Card>
  );
}
