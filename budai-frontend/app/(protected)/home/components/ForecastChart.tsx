"use client";

import React, { useMemo } from "react";
import { Card, Chip } from "@heroui/react";

export interface TrajectoryPoint {
  day: number;
  date: string;
  median: number;
  lower_bound: number;
  upper_bound: number;
}

interface ForecastChartProps {
  trajectory: TrajectoryPoint[];
  safetyBufferFloor?: number;
  timeWindow?: string;
}

export function ForecastChart({
  trajectory = [],
  safetyBufferFloor = 800.0,
  timeWindow = "60D Forecast",
}: ForecastChartProps) {
  const chartData = useMemo(() => {
    if (!trajectory || trajectory.length === 0) {
      return {
        polygonPoints:
          "0,130 50,115 100,100 150,85 200,105 250,90 300,75 350,65 400,50 450,45 500,35 550,30 600,20 600,85 550,95 500,105 450,115 400,120 350,135 300,140 250,150 200,165 150,145 100,155 50,165 0,170",
        pathD: "M 0,150 Q 50,135 100,120 T 200,135 T 300,105 T 400,85 T 500,70 T 600,50",
        yFloor: 150,
        dateLabels: ["Nov 4", "Nov 18", "Dec 02", "Dec 16", "Dec 30", "Jan 13"],
      };
    }

    const minVal = Math.min(...trajectory.map((p) => p.lower_bound), safetyBufferFloor * 0.7);
    const maxVal = Math.max(...trajectory.map((p) => p.upper_bound), safetyBufferFloor * 2.5);
    const range = maxVal - minVal || 1;

    const getY = (val: number) => {
      const normalized = (val - minVal) / range;
      return Math.round(180 - normalized * 150);
    };

    const count = trajectory.length;
    const upperPts: string[] = [];
    const lowerPts: string[] = [];
    const medianPts: { x: number; y: number }[] = [];

    trajectory.forEach((pt, i) => {
      const x = Math.round((i / (count - 1)) * 600);
      const yUpper = getY(pt.upper_bound);
      const yLower = getY(pt.lower_bound);
      const yMed = getY(pt.median);

      upperPts.push(`${x},${yUpper}`);
      lowerPts.unshift(`${x},${yLower}`);
      medianPts.push({ x, y: yMed });
    });

    const polygonPoints = `${upperPts.join(" ")} ${lowerPts.join(" ")}`;

    let pathD = `M ${medianPts[0].x},${medianPts[0].y}`;
    for (let i = 1; i < medianPts.length; i++) {
      const prev = medianPts[i - 1];
      const curr = medianPts[i];
      const cx = (prev.x + curr.x) / 2;
      pathD += ` Q ${prev.x},${prev.y} ${cx},${(prev.y + curr.y) / 2}`;
    }
    const last = medianPts[medianPts.length - 1];
    pathD += ` T ${last.x},${last.y}`;

    const yFloor = getY(safetyBufferFloor);

    const step = Math.max(1, Math.floor(count / 5));
    const dateLabels: string[] = [];
    for (let i = 0; i < count; i += step) {
      dateLabels.push(trajectory[i].date);
      if (dateLabels.length === 6) break;
    }

    return {
      polygonPoints,
      pathD,
      yFloor,
      dateLabels,
    };
  }, [trajectory, safetyBufferFloor]);

  return (
    <Card className="glass-card rounded-2xl p-5 border border-slate-800 shadow-none">
      {/* Chart Header */}
      <div className="flex items-center justify-between mb-4 flex-wrap gap-2 w-full">
        <div>
          <div className="flex items-center space-x-2">
            <h2 className="text-sm font-semibold text-white">Cash Flow Projection & Buffer Range</h2>
            <Chip size="sm" variant="soft" color="accent" className="font-mono text-[11px]">
              {timeWindow} Forward Model
            </Chip>
          </div>
          <p className="text-xs text-slate-400 mt-0.5">
            Calculated using recurring cadence, scheduled bills, and discretionary spending variance
          </p>
        </div>
        <div className="flex items-center space-x-3 text-xs">
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-cyan-400" />
            <span className="text-slate-300">Expected</span>
          </div>
          <div className="flex items-center space-x-1.5">
            <span className="w-2.5 h-2.5 rounded-sm bg-cyan-500/20 border border-cyan-500/40" />
            <span className="text-slate-400">95% Range</span>
          </div>
        </div>
      </div>

      {/* SVG Trajectory Chart */}
      <div className="relative w-full h-48 bg-slate-900/60 rounded-xl p-2 border border-slate-800/80 overflow-hidden">
        <svg className="w-full h-full" viewBox="0 0 600 200" preserveAspectRatio="none">
          <defs>
            <linearGradient id="confidenceBand" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="#06b6d4" stopOpacity="0.3" />
              <stop offset="100%" stopColor="#06b6d4" stopOpacity="0.03" />
            </linearGradient>
            <linearGradient id="lineGlow" x1="0" y1="0" x2="1" y2="0">
              <stop offset="0%" stopColor="#38bdf8" />
              <stop offset="100%" stopColor="#06b6d4" />
            </linearGradient>
          </defs>

          {/* Gridlines */}
          <line x1="0" y1="40" x2="600" y2="40" stroke="#1e293b" strokeDasharray="4" />
          <line x1="0" y1="80" x2="600" y2="80" stroke="#1e293b" strokeDasharray="4" />
          <line x1="0" y1="120" x2="600" y2="120" stroke="#1e293b" strokeDasharray="4" />
          <line x1="0" y1="160" x2="600" y2="160" stroke="#1e293b" strokeDasharray="4" />

          {/* Safety Buffer Floor Line */}
          <line
            x1="0"
            y1={chartData.yFloor}
            x2="600"
            y2={chartData.yFloor}
            stroke="#f59e0b"
            strokeWidth="1.5"
            strokeDasharray="5 3"
            opacity="0.85"
          />
          <text
            x="420"
            y={Math.max(16, chartData.yFloor - 6)}
            fill="#f59e0b"
            fontSize="11"
            fontWeight="500"
            className="font-mono"
          >
            Safety Buffer Floor: £{safetyBufferFloor.toFixed(2)}
          </text>

          {/* 95% Confidence Band Polygon */}
          <polygon points={chartData.polygonPoints} fill="url(#confidenceBand)" />

          {/* Median Trajectory Line */}
          <path
            d={chartData.pathD}
            fill="none"
            stroke="url(#lineGlow)"
            strokeWidth="3"
            strokeLinecap="round"
          />

          {/* Today Marker */}
          <circle cx="20" cy="150" r="4.5" fill="#38bdf8" stroke="#0b0f17" strokeWidth="2" />
        </svg>

        {/* Date Markers */}
        <div className="absolute bottom-1.5 left-3 right-3 text-[11px] text-slate-500 font-mono flex justify-between">
          {chartData.dateLabels.map((dateStr, idx) => (
            <span key={idx}>{dateStr}</span>
          ))}
        </div>
      </div>
    </Card>
  );
}
