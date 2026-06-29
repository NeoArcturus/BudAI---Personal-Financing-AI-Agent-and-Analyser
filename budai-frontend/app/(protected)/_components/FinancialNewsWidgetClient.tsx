"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { Newspaper, ExternalLink, Image as ImageIcon } from "lucide-react";
import { Card, Skeleton, ScrollShadow, Text, Link } from "@heroui/react";
import WidgetFlipCard from "./WidgetFlipCard";
import { useRouter } from "next/navigation";
import { useBudAI } from "@/app/context/AppContext";
import Image from "next/image";

interface NewsItem {
  title: string;
  snippet: string;
  url: string;
  source?: string;
  image_url?: string;
}

interface FinancialNewsWidgetProps {
  initialNews?: { geopolitical: NewsItem[]; market: NewsItem[] };
}

export default function FinancialNewsWidgetClient({
  initialNews,
}: FinancialNewsWidgetProps) {
  const router = useRouter();
  const { createNewSession } = useBudAI();

  const { data: newsData, isLoading: isNewsLoading } = useQuery<{
    geopolitical: NewsItem[];
    market: NewsItem[];
  }>({
    queryKey: ["market-news"],
    queryFn: async () => {
      const res = await apiFetch("/api/market/news", {}, true);
      return (await res.json()) as { geopolitical: NewsItem[]; market: NewsItem[] };
    },
    initialData: initialNews,
    staleTime: 7200000,
  });

  const handleDiscuss = () => {
    const sessionId = createNewSession(
      `Financial & Geopolitical News Analysis`,
      {
        type: "market_audit",
        data: {
          news: JSON.stringify(newsData || {}),
        },
      },
    );
    router.push(`/advisor?session=${sessionId}`);
  };

  const renderNewsSection = (title: string, items: NewsItem[] = []) => (
    <div className="space-y-6">
      <div className="px-1 flex items-center justify-between">
        <Text className="text-[9px] font-black uppercase tracking-[0.4em] text-primary/50 italic">
          {title}
        </Text>
        <div className="h-[0.5px] flex-1 bg-white/5 ml-4" />
      </div>
      {items.length === 0 ? (
        <Text className="text-foreground/20 text-[10px] font-black uppercase tracking-widest px-1">
          No content available
        </Text>
      ) : (
        items.map((item, i) => (
          <div
            key={`${title}-${i}`}
            className="p-6 rounded-xl bg-white/3 backdrop-blur-3xl border-[0.5px] border-white/10 shadow-inner hover:border-primary/40 transition-all group relative"
          >
            <Link
              href={item.url}
              className="flex gap-6 items-start p-0 h-auto w-full text-left"
            >
              {}
              <div className="w-16 h-16 rounded-lg overflow-hidden shrink-0 bg-white/5 border-[0.5px] border-white/10 flex items-center justify-center shadow-sm">
                {item.image_url ? (
                  <Image
                    unoptimized
                    alt={item.title}
                    className="w-full h-full object-cover group-hover:scale-110 transition-transform duration-700"
                    src={item.image_url}
                    width={2}
                    height={2}
                  />
                ) : (
                  <ImageIcon size={20} className="text-foreground/10" />
                )}
              </div>

              {}
              <div className="flex-1 flex flex-col items-start min-w-0">
                <div className="flex items-center justify-between w-full mb-2">
                  <Text className="text-[9px] font-black uppercase tracking-[0.2em] text-primary font-mono">
                    {item.source?.toUpperCase() || "INTEL_SOURCE"}
                  </Text>
                </div>
                <Text className="text-foreground text-[14px] font-black leading-snug tracking-tight uppercase italic group-hover:text-primary transition-colors line-clamp-2">
                  {item.title}
                </Text>
                <Text className="text-foreground/30 text-[10px] font-medium leading-relaxed line-clamp-1 mt-2 tracking-wide uppercase">
                  {item.snippet}
                </Text>
                <div className="flex items-center gap-2 mt-3 text-primary opacity-0 group-hover:opacity-100 transition-all -translate-x-2 group-hover:translate-x-0">
                  <Text className="text-[8px] font-black uppercase tracking-[0.3em]">
                    Read here
                  </Text>
                  <ExternalLink size={10} />
                </div>
              </div>
            </Link>
          </div>
        ))
      )}
    </div>
  );

  return (
    <WidgetFlipCard
      insight={undefined}
      isLoading={false}
      isDataLoading={isNewsLoading}
      onDiscuss={handleDiscuss}
    >
      <Card className="w-full h-full liquid-glass border-none rounded-xl flex flex-col relative overflow-hidden">
        <Card.Header className="p-8 border-b-[0.5px] border-white/5 shrink-0 flex items-center justify-between z-10">
          <div className="flex items-center gap-4">
            <div className="w-10 h-10 rounded-xl bg-primary/10 border-[0.5px] border-primary/20 flex items-center justify-center text-primary shadow-[0_0_20px_rgba(0,127,255,0.05)]">
              <Newspaper size={20} />
            </div>
            <div className="flex flex-col">
              <h3 className="text-[10px] font-black text-foreground uppercase tracking-[0.4em] italic m-0">
                Global news
              </h3>
              <p className="text-primary/50 text-[8px] font-black uppercase tracking-[0.3em] mt-1.5">
                Live news feed
              </p>
            </div>
          </div>
        </Card.Header>

        <Card.Content className="p-0 flex-1 flex flex-col gap-3 overflow-hidden min-h-0 relative">
          <div className="absolute inset-0 bg-[radial-gradient(circle_at_50%_50%,rgba(0,127,255,0.01)_0%,transparent_70%)] pointer-events-none" />
          <ScrollShadow
            hideScrollBar
            className="flex-1 min-h-0 overflow-y-auto space-y-12 p-8 scrollbar-hide [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] relative z-10"
          >
            {isNewsLoading ? (
              Array(4)
                .fill(0)
                .map((_, i) => (
                  <div
                    key={i}
                    className="p-6 rounded-xl bg-white/5 border-[0.5px] border-white/10 space-y-4 shadow-sm"
                  >
                    <Skeleton
                      animationType="shimmer"
                      className="h-2 w-3/4 rounded bg-white/5"
                    />
                    <Skeleton
                      animationType="shimmer"
                      className="h-2 w-full rounded bg-white/5"
                    />
                  </div>
                ))
            ) : (
              <>
                {renderNewsSection("Geopolitical News", newsData?.geopolitical)}
                {renderNewsSection("Market News", newsData?.market)}
              </>
            )}
          </ScrollShadow>
          <div className="p-4 border-t-[0.5px] border-white/5 bg-white/1 relative z-10">
            <Text className="text-center text-[8px] text-foreground/20 font-black uppercase tracking-[0.5em]">
              Real time news stream
            </Text>
          </div>
        </Card.Content>
      </Card>
    </WidgetFlipCard>
  );
}
