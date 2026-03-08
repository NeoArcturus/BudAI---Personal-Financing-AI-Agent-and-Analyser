"use client";

import React, { useEffect, useRef } from "react";
// chart.js/auto automatically registers all necessary controllers (like "line"), scales, and the Filler plugin
import Chart from "chart.js/auto";
import { ChartConfiguration } from "chart.js";

interface DynamicChartProps {
  config: ChartConfiguration;
}

export default function DynamicChart({ config }: DynamicChartProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    // Abort if the canvas isn't mounted or config is missing
    if (!canvasRef.current || !config) return;

    // Destroy the previous chart instance before rendering a new one to prevent canvas overlap
    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    // Initialize the pure native Chart.js instance
    chartInstance.current = new Chart(canvasRef.current, config);

    // Cleanup function when the component unmounts or config changes
    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [config]);

  if (!config) return null;

  return (
    // The relative wrapper with min-h-[300px] ensures the canvas never collapses to 0 pixels
    <div className="relative w-full h-full min-h-75 p-6 bg-[#161B22] rounded-3xl border border-slate-800 shadow-2xl animate-in fade-in zoom-in-95">
      <canvas ref={canvasRef} className="w-full h-full"></canvas>
    </div>
  );
}
