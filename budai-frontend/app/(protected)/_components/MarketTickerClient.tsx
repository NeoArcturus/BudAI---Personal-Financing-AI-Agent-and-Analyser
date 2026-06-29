"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { TrendingUp, TrendingDown, Minus } from "lucide-react";
import { cn } from "@/lib/utils";

interface MarketTickerItem {
  symbol: string;
  price: number;
  change: number;
}

interface MarketTickerProps {
  initialData?: MarketTickerItem[];
}

export default function MarketTickerClient({ initialData }: MarketTickerProps) {
  const { data, isLoading } = useQuery<MarketTickerItem[]>({
    queryKey: ["market-ticker"],
    queryFn: async () => {
      const res = await apiFetch("/api/market/ticker", {}, true);
      const json = (await res.json()) as { tickers: MarketTickerItem[] };
      return json.tickers || [];
    },
    initialData,
    refetchInterval: 1000 * 60 * 15,
  });

  if ((isLoading && !initialData) || !data || data.length === 0) return null;

  return (
    <div className="w-full h-8 bg-black/20 backdrop-blur-md border-b-[0.5px] border-white/10 flex items-center overflow-hidden shrink-0 relative z-20">
      <div className="flex animate-marquee whitespace-nowrap gap-16 px-8 w-max">
        {data.map((item: MarketTickerItem, i: number) => (
          <div key={i} className="flex items-center gap-3">
            <span className="text-foreground/30 font-black text-[9px] tracking-[0.3em] uppercase italic">
              {item.symbol.replace("=X", "").replace("=F", "").replace("^", "")}
            </span>
            <span className="text-foreground font-mono text-[11px] font-bold">
              {item.price.toFixed(2)}
            </span>
            <div className={cn(
              "flex items-center gap-1 text-[9px] font-black tracking-tighter font-mono",
              item.change > 0 ? "text-green-500" : item.change < 0 ? "text-primary" : "text-foreground/20"
            )}>
              {item.change > 0 ? <TrendingUp size={12} /> : item.change < 0 ? <TrendingDown size={12} /> : <Minus size={12} />}
              {Math.abs(item.change).toFixed(2)}%
            </div>
          </div>
        ))}
        {}
        {data.map((item: MarketTickerItem, i: number) => (
          <div key={`dup-${i}`} className="flex items-center gap-3">
            <span className="text-foreground/30 font-black text-[9px] tracking-[0.3em] uppercase italic">
              {item.symbol.replace("=X", "").replace("=F", "").replace("^", "")}
            </span>
            <span className="text-foreground font-mono text-[11px] font-bold">
              {item.price.toFixed(2)}
            </span>
            <div className={cn(
              "flex items-center gap-1 text-[9px] font-black tracking-tighter font-mono",
              item.change > 0 ? "text-green-500" : item.change < 0 ? "text-primary" : "text-foreground/20"
            )}>
              {item.change > 0 ? <TrendingUp size={12} /> : item.change < 0 ? <TrendingDown size={12} /> : <Minus size={12} />}
              {Math.abs(item.change).toFixed(2)}%
            </div>
          </div>
        ))}
      </div>
      <style jsx global>{`
        @keyframes marquee {
          0% { transform: translateX(0%); }
          100% { transform: translateX(-50%); }
        }
        .animate-marquee {
          animation: marquee 30s linear infinite;
        }
        .animate-marquee:hover {
          animation-play-state: paused;
        }
      `}</style>
    </div>
  );
}
