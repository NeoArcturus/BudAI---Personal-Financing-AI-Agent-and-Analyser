"use client";

import React, { useEffect, useRef } from "react";
import Chart from "chart.js/auto";
import { ChartConfiguration } from "chart.js";

interface DynamicChartProps {
  config: ChartConfiguration;
}

export default function DynamicChart({ config }: DynamicChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current || !config) return;

    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    chartInstance.current = new Chart(canvasRef.current, config);

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [config]);

  if (!config) return null;

  return (
    <div className="relative w-full h-87.5 flex items-center justify-center">
      <canvas ref={canvasRef}></canvas>
    </div>
  );
}
