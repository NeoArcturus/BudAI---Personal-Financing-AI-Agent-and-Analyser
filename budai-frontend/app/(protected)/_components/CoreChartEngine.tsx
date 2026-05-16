
"use client";

import React, { useEffect, useRef } from "react";
import Chart from "chart.js/auto";
import { ChartConfiguration, ChartOptions, Plugin } from "chart.js";

interface CoreChartEngineProps {
  config: ChartConfiguration | null;
}

const customGlowPlugin: Plugin = {
  id: "customGlow",
  beforeDatasetsDraw: (chart) => {
    const ctx = chart.ctx;
    ctx.save();
    const activeDataset = chart.data.datasets[0];
    const config = chart.config as ChartConfiguration;

    if (
      activeDataset &&
      (activeDataset.type === "line" || config.type === "line")
    ) {
      const color = activeDataset.borderColor;
      ctx.shadowColor =
        typeof color === "string" ? color : "rgba(61, 115, 255, 0.5)";
      ctx.shadowBlur = 12;
      ctx.shadowOffsetX = 0;
      ctx.shadowOffsetY = 4;
    }
  },
  afterDatasetsDraw: (chart) => {
    chart.ctx.restore();
  },
};

export default function CoreChartEngine({ config }: CoreChartEngineProps) {
  const canvasRef = useRef<HTMLCanvasElement | null>(null);
  const chartInstance = useRef<Chart | null>(null);

  useEffect(() => {
    if (!canvasRef.current || !config) return;

    if (chartInstance.current) {
      chartInstance.current.destroy();
    }

    const baseOptions: ChartOptions = {
      responsive: true,
      maintainAspectRatio: false,
      animation: {
        duration: 600,
        easing: "easeOutQuart",
      },
      plugins: {
        legend: {
          display: false,
        },
        tooltip: {
          backgroundColor: "rgba(26, 28, 36, 0.9)",
          titleColor: "#FFFFFF",
          bodyColor: "#8B8E98",
          borderColor: "rgba(255,255,255,0.1)",
          borderWidth: 1,
          padding: 12,
          boxPadding: 6,
          usePointStyle: true,
        },
      },
      scales: {
        x: {
          grid: { display: false },
          border: { display: false },
          ticks: {
            color: "#8B8E98",
            font: { size: 11, family: "sans-serif" },
            padding: 10,
          },
        },
        y: {
          grid: { color: "rgba(255,255,255,0.03)", drawTicks: false },
          border: { display: false },
          ticks: {
            color: "#8B8E98",
            font: { size: 11, family: "sans-serif" },
            padding: 10,
          },
        },
      },
      interaction: {
        mode: "index",
        intersect: false,
      },
    };

    const enhancedDatasets =
      config.data?.datasets.map((dataset) => {
        const isLine = dataset.type === "line" || config.type === "line";
        const isBar = dataset.type === "bar" || config.type === "bar";

        return {
          ...dataset,
          ...(isLine && {
            tension: 0.4,
            borderWidth: 3,
            pointRadius: 0,
            pointHoverRadius: 6,
          }),
          ...(isBar && {
            borderRadius: 6,
            borderSkipped: false,
          }),
        };
      }) || [];

    const isCircular =
      config.type === "pie" ||
      config.type === "doughnut" ||
      config.type === "radar" ||
      (config.data?.datasets &&
        config.data.datasets.some(
          (d) =>
            d.type === "pie" || d.type === "doughnut" || d.type === "radar",
        ));

    const finalConfig: ChartConfiguration = {
      ...config,
      data: {
        ...config.data,
        datasets: enhancedDatasets,
      },
      options: {
        ...config.options,
        ...baseOptions,
        plugins: {
          ...config.options?.plugins,
          ...baseOptions.plugins,
        },
        scales: isCircular
          ? {
              ...config.options?.scales,
              x: { display: false },
              y: { display: false },
            }
          : {
              ...config.options?.scales,
              ...baseOptions.scales,
            },
      },
      plugins: [customGlowPlugin],
    };

    chartInstance.current = new Chart(canvasRef.current, finalConfig);

    return () => {
      if (chartInstance.current) {
        chartInstance.current.destroy();
      }
    };
  }, [config]);

  if (!config) return null;

  return (
    <div className="relative w-full h-full min-h-0 flex items-center justify-center overflow-hidden">
      <canvas ref={canvasRef} className="w-full h-full"></canvas>
    </div>
  );
}
