import { NativeChartConfig, ToolParameters, BankChartData } from "@/types";
import { ChartDataset } from "chart.js";

interface ChartAnimationContext {
  type: string;
  mode: string;
  dataIndex: number;
  datasetIndex: number;
  index: number;
  chart: {
    scales: {
      y: {
        getPixelForValue: (value: number) => number;
      };
    };
    getDatasetMeta: (index: number) => {
      data: {
        getProps: (props: string[], final: boolean) => { y: number };
      }[];
    };
  };
}

interface ProgressiveAnimationContext extends ChartAnimationContext {
  xStarted?: boolean;
  yStarted?: boolean;
}

const baseOptions = {
  responsive: true,
  maintainAspectRatio: false,
  animation: {
    duration: 1000,
    easing: "easeOutQuart" as const,
  },
  interaction: {
    mode: "index" as const,
    intersect: false,
  },
  plugins: {
    legend: {
      position: "top" as const,
      labels: {
        color: "#94a3b8",
        font: { family: "monospace" },
        usePointStyle: true,
        boxWidth: 8,
      },
    },
    tooltip: {
      backgroundColor: "rgba(13, 17, 23, 0.95)",
      titleColor: "#00FFAA",
      bodyColor: "#e2e8f0",
      borderColor: "#1e293b",
      borderWidth: 1,
      padding: 12,
      usePointStyle: true,
    },
  },
  scales: {
    y: {
      grace: "5%",
      grid: { color: "#1e293b", drawBorder: false },
      ticks: { color: "#94a3b8", font: { family: "monospace" } },
    },
    x: {
      grid: { display: false, drawBorder: false },
      ticks: { color: "#94a3b8", maxTicksLimit: 12, maxRotation: 0 },
    },
  },
};

// Extended 16-color palette to prevent repetition in charts with many categories
const colorPalette = [
  "#00FFAA", // Bright Green
  "#3b82f6", // Blue
  "#ef4444", // Red
  "#f59e0b", // Yellow/Amber
  "#a855f7", // Purple
  "#ec4899", // Pink
  "#14b8a6", // Teal
  "#f97316", // Orange
  "#6366f1", // Cyan
  "#8b5cf6", // Violet
  "#84cc16", // Emerald
  "#eab308", // Lime
  "#0ea5e9", // Indigo
  "#d946ef", // Fuchsia
  "#10b981", // Light Teal
  "#f43f5e", // Rose
];

const getColorForMetric = (
  metricName: string,
  defaultIndex: number,
): string => {
  const lower = metricName.toLowerCase();
  if (
    lower.includes("balance") ||
    lower.includes("flow") ||
    lower.includes("optimal")
  )
    return "#00FFAA";
  if (
    lower.includes("expense") ||
    lower.includes("spend") ||
    lower.includes("careless")
  )
    return "#ef4444";
  if (lower.includes("income")) return "#3b82f6";
  return colorPalette[defaultIndex % colorPalette.length];
};

const getProgressiveAnimation = (dataLength: number) => {
  const totalDuration = 2000;
  const delayBetweenPoints = totalDuration / Math.max(1, dataLength);

  return {
    x: {
      type: "number",
      easing: "linear",
      duration: delayBetweenPoints,
      from: NaN,
      delay(ctx: ProgressiveAnimationContext) {
        if (ctx.type !== "data" || ctx.xStarted) {
          return 0;
        }
        ctx.xStarted = true;
        return ctx.index * delayBetweenPoints;
      },
    },
    y: {
      type: "number",
      easing: "linear",
      duration: delayBetweenPoints,
      from(ctx: ProgressiveAnimationContext) {
        return ctx.index === 0
          ? ctx.chart.scales.y.getPixelForValue(0)
          : ctx.chart
              .getDatasetMeta(ctx.datasetIndex)
              .data[ctx.index - 1].getProps(["y"], true).y;
      },
      delay(ctx: ProgressiveAnimationContext) {
        if (ctx.type !== "data" || ctx.yStarted) {
          return 0;
        }
        ctx.yStarted = true;
        return ctx.index * delayBetweenPoints;
      },
    },
  };
};

const getDelayedAnimation = () => {
  let delayed = false;
  return {
    duration: 1000,
    easing: "easeOutQuart",
    onComplete: () => {
      delayed = true;
    },
    delay: (context: ChartAnimationContext) => {
      let delay = 0;
      if (context.type === "data" && context.mode === "default" && !delayed) {
        delay = context.dataIndex * 50 + context.datasetIndex * 50;
      }
      return delay;
    },
  };
};

export const buildChartConfig = (
  type: string,
  payloadData: BankChartData[],
  params: ToolParameters,
  customTitle?: string,
): NativeChartConfig | null => {
  if (type === "categorized") {
    const allLabels = Array.from(
      new Set(
        payloadData.flatMap((b) => b.data.map((d) => String(d.Category))),
      ),
    );
    const datasets = payloadData.map((b, i) => {
      const metricColor =
        payloadData.length > 1
          ? colorPalette[i % colorPalette.length]
          : getColorForMetric("expense", i);

      return {
        label: `${b.bank_name} Spent (£)`,
        data: allLabels.map((l) => {
          const pt = b.data.find((d) => String(d.Category) === l);
          return pt ? Number(pt.Total_Amount) : 0;
        }),
        backgroundColor: metricColor,
        borderRadius: 6,
        hoverBackgroundColor: `${metricColor}cc`,
      };
    });

    return {
      type: "bar",
      data: { labels: allLabels, datasets },
      options: {
        ...baseOptions,
        animation: getDelayedAnimation() as unknown as Record<string, unknown>,
        plugins: {
          ...baseOptions.plugins,
          title: {
            display: true,
            text:
              customTitle ||
              `Spending Breakdown (${params.from_date} to ${params.to_date})`,
            color: "#ffffff",
            font: { size: 16 },
          },
        },
      },
    } as unknown as NativeChartConfig;
  }

  if (type === "categorized_doughnut") {
    const agg: Record<string, number> = {};
    payloadData.forEach((b) => {
      b.data.forEach((d) => {
        const cat = String(d.Category);
        if (cat.toLowerCase() !== "income") {
          agg[cat] = (agg[cat] || 0) + Number(d.Total_Amount);
        }
      });
    });
    const labels = Object.keys(agg);
    const amounts = Object.values(agg);

    return {
      type: "doughnut",
      data: {
        labels,
        datasets: [
          {
            data: amounts,
            backgroundColor: colorPalette,
            borderColor: "#161B22",
            borderWidth: 4,
            hoverOffset: 12,
          },
        ],
      },
      options: {
        ...baseOptions,
        cutout: "75%",
        animation: getDelayedAnimation() as unknown as Record<string, unknown>,
        scales: { x: { display: false }, y: { display: false } },
        plugins: {
          ...baseOptions.plugins,
          title: {
            display: true,
            text:
              customTitle ||
              `Expense Distribution (${params.from_date} to ${params.to_date})`,
            color: "#ffffff",
            font: { size: 16 },
          },
        },
      },
    } as unknown as NativeChartConfig;
  }

  if (type === "cash_flow_mixed") {
    const allMonths = Array.from(
      new Set(payloadData.flatMap((b) => b.data.map((d) => String(d.Month)))),
    ).sort();
    const netBalance = allMonths.map((m) =>
      payloadData.reduce(
        (s, b) =>
          s + Number(b.data.find((d) => d.Month === m)?.Net_Balance || 0),
        0,
      ),
    );
    const income = allMonths.map((m) =>
      payloadData.reduce(
        (s, b) => s + Number(b.data.find((d) => d.Month === m)?.Income || 0),
        0,
      ),
    );
    const expense = allMonths.map((m) =>
      payloadData.reduce(
        (s, b) => s + Number(b.data.find((d) => d.Month === m)?.Expense || 0),
        0,
      ),
    );

    return {
      type: "bar",
      data: {
        labels: allMonths,
        datasets: [
          {
            type: "line",
            label: "Net Flow (£)",
            data: netBalance,
            borderColor: getColorForMetric("balance", 0),
            backgroundColor: `${getColorForMetric("balance", 0)}1A`,
            borderWidth: 3,
            tension: 0.4,
            fill: true,
            pointBackgroundColor: getColorForMetric("balance", 0),
            pointRadius: 3,
            pointHoverRadius: 6,
          },
          {
            type: "bar",
            label: "Income (£)",
            data: income,
            backgroundColor: getColorForMetric("income", 1),
            borderRadius: 4,
          },
          {
            type: "bar",
            label: "Expenses (£)",
            data: expense,
            backgroundColor: getColorForMetric("expense", 2),
            borderRadius: 4,
          },
        ],
      },
      options: {
        ...baseOptions,
        animation: getDelayedAnimation() as unknown as Record<string, unknown>,
        plugins: {
          ...baseOptions.plugins,
          title: {
            display: true,
            text: customTitle || "Income vs Expense Matrix",
            color: "#ffffff",
            font: { size: 16 },
          },
        },
      },
    } as unknown as NativeChartConfig;
  }

  if (type === "health_radar") {
    const labels = payloadData[0]?.data.map((d) => String(d.Metric)) || [];
    const scores = payloadData[0]?.data.map((d) => Number(d.Score)) || [];

    return {
      type: "radar",
      data: {
        labels,
        datasets: [
          {
            label: "Health Index",
            data: scores,
            backgroundColor: "rgba(0, 255, 170, 0.2)",
            borderColor: "#00FFAA",
            borderWidth: 2,
            pointBackgroundColor: "#161B22",
            pointBorderColor: "#00FFAA",
            pointHoverBackgroundColor: "#fff",
            pointHoverBorderColor: "#00FFAA",
            pointHoverRadius: 6,
          },
        ],
      },
      options: {
        ...baseOptions,
        animation: getDelayedAnimation() as unknown as Record<string, unknown>,
        scales: {
          x: { display: false },
          y: { display: false },
          r: {
            angleLines: { color: "#1e293b" },
            grid: { color: "#1e293b" },
            pointLabels: {
              color: "#94a3b8",
              font: { family: "monospace", size: 11 },
            },
            ticks: { display: false, min: 0, max: 100, stepSize: 20 },
          },
        },
        plugins: {
          ...baseOptions.plugins,
          title: {
            display: true,
            text: customTitle || "Financial Health Profile",
            color: "#ffffff",
            font: { size: 16 },
          },
        },
      },
    } as unknown as NativeChartConfig;
  }

  if (type === "expense_forecast" || type === "balance_forecast") {
    const allDays = Array.from(
      new Set(payloadData.flatMap((b) => b.data.map((d) => String(d.Day)))),
    ).sort((a, b) => Number(a.split(" ")[1]) - Number(b.split(" ")[1]));
    const datasets: ChartDataset<"line">[] = [];

    if (type === "balance_forecast") {
      payloadData.forEach((b, i) => {
        const metricColor =
          payloadData.length > 1
            ? colorPalette[i % colorPalette.length]
            : getColorForMetric("balance", i);
        datasets.push({
          label: `${b.bank_name} Expected`,
          data: allDays.map((day) =>
            Number(
              b.data.find((d) => d.Day === day)?.["Expected Balance"] || 0,
            ),
          ),
          borderColor: metricColor,
          backgroundColor: `${metricColor}1A`,
          fill: false,
          tension: 0.4,
          pointRadius: 0,
          pointHitRadius: 10,
          pointHoverRadius: 5,
        });
      });
    } else {
      payloadData.forEach((b, i) => {
        const metricColor =
          payloadData.length > 1
            ? colorPalette[i % colorPalette.length]
            : getColorForMetric("spend", i);
        datasets.push({
          label: `${b.bank_name} Spend`,
          data: allDays.map((day) =>
            Number(
              b.data.find((d) => d.Day === day)?.[
                "Projected Daily Spend (£)"
              ] || 0,
            ),
          ),
          borderColor: metricColor,
          backgroundColor: `${metricColor}1A`,
          fill: true,
          tension: 0.4,
          pointRadius: 0,
          pointHitRadius: 10,
          pointHoverRadius: 5,
        });
      });
    }

    return {
      type: "line",
      data: { labels: allDays, datasets },
      options: {
        ...baseOptions,
        animation: getProgressiveAnimation(allDays.length) as unknown as Record<
          string,
          unknown
        >,
        plugins: {
          ...baseOptions.plugins,
          title: {
            display: true,
            text: customTitle || `AI Forecast Analysis (${params.days} Days)`,
            color: "#ffffff",
            font: { size: 16 },
          },
        },
      },
    } as unknown as NativeChartConfig;
  }

  if (type.startsWith("historical")) {
    const timeType = type.split("_")[1] || "monthly";
    const allDates = Array.from(
      new Set(payloadData.flatMap((b) => b.data.map((d) => String(d.Date)))),
    ).sort();
    const datasets: ChartDataset<"line">[] = [];

    payloadData.forEach((b, idx) => {
      const metricColor =
        payloadData.length > 1
          ? colorPalette[idx % colorPalette.length]
          : getColorForMetric("expense", idx);

      datasets.push({
        label: b.bank_name,
        data: allDates.map((date) =>
          Number(b.data.find((d) => d.Date === date)?.Amount || 0),
        ),
        borderColor: metricColor,
        fill: false,
        tension: 0.4,
        pointRadius: 2,
        pointBackgroundColor: metricColor,
        pointHitRadius: 10,
        pointHoverRadius: 6,
      });
    });

    return {
      type: "line",
      data: { labels: allDates, datasets },
      options: {
        ...baseOptions,
        animation: getProgressiveAnimation(
          allDates.length,
        ) as unknown as Record<string, unknown>,
        plugins: {
          ...baseOptions.plugins,
          title: {
            display: true,
            text:
              customTitle ||
              `${timeType.charAt(0).toUpperCase() + timeType.slice(1)} Expense Analysis`,
            color: "#ffffff",
            font: { size: 16 },
          },
        },
      },
    } as unknown as NativeChartConfig;
  }

  return null;
};
