"use client";

import React from "react";
import { useQuery } from "@tanstack/react-query";
import { Card, Skeleton, Spinner } from "@heroui/react";
import { apiFetch } from "@/lib/api";
import { DynamicChart } from "@/app/(protected)/advisor/components/DynamicChart";
import { motion } from "framer-motion";

export interface WidgetLayout {
  x: number;
  y: number;
  width: number;
  height: number;
}

export interface WidgetConfiguration {
  widget_uuid: string;
  type: string;
  title: string;
  layout: WidgetLayout;
  query_payload: {
    tool_name: string;
    parameters: any;
  };
}

interface UniversalWidgetContainerProps {
  config: WidgetConfiguration;
}

export const UniversalWidgetContainer: React.FC<UniversalWidgetContainerProps> = ({ config }) => {
  const { data, isLoading, isError, error } = useQuery({
    queryKey: ["widget-data", config.widget_uuid, config.query_payload],
    queryFn: async () => {
      const response = await apiFetch("/api/dashboard/widgets/data", {
        method: "POST",
        body: JSON.stringify(config.query_payload),
      }, true);
      if (!response.ok) {
        throw new Error("Failed to fetch widget data");
      }
      return (await response.json()) as any as {
        summary?: string;
        chart_data?: {
          labels: string[];
          datasets: any[];
        };
      };
    },
    // Only refetch if the payload changes, or poll if configured
    staleTime: 1000 * 60 * 5, 
  });

  return (
    <motion.div
      layout
      initial={{ opacity: 0, scale: 0.95 }}
      animate={{ opacity: 1, scale: 1 }}
      exit={{ opacity: 0, scale: 0.95 }}
      className="w-full h-full min-h-[300px] flex flex-col"
    >
      <div className="w-full h-full relative flex-1">
        {isLoading ? (
          <div className="w-full h-full flex items-center justify-center liquid-glass rounded-xl">
            <Spinner size="lg" color="current" className="text-primary" />
          </div>
        ) : isError ? (
          <div className="w-full h-full flex items-center justify-center text-danger text-xs uppercase tracking-wider font-bold text-center liquid-glass rounded-xl">
            Failed to load data
            <br />
            <span className="text-[10px] font-mono mt-2 opacity-50">{String(error)}</span>
          </div>
        ) : (
          <DynamicChart 
            hideWrapper={true}
            config={{
              type: config.type as any || "bar",
              title: config.title,
              labels: data?.chart_data?.labels || [],
              datasets: data?.chart_data?.datasets || [],
              options: {
                showLegend: true
              }
            }} 
          />
        )}
      </div>
    </motion.div>
  );
};
