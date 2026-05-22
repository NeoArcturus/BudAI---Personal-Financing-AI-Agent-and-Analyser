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
    duration: 600,
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
        color: "rgba(255, 255, 255, 0.7)",
        font: { family: "var(--font-jakarta), sans-serif" },
        usePointStyle: true,
        boxWidth: 8,
      },
    },
    tooltip: {
      backgroundColor: "var(--popover)",
      titleColor: "var(--primary)",
      bodyColor: "#FFFFFF",
      borderColor: "rgba(255, 255, 255, 0.1)",
      borderWidth: 1,
      padding: 12,
      usePointStyle: true,
    },
  },
  scales: {
    y: {
      grace: "5%",
      grid: { color: "rgba(255, 255, 255, 0.05)", drawBorder: false },
      ticks: {
        color: "rgba(255, 255, 255, 0.8)",
        font: { family: "var(--font-jakarta), sans-serif" },
      },
    },
    x: {
      grid: { display: false, drawBorder: false },
      ticks: {
        color: "rgba(255, 255, 255, 0.8)",
        maxTicksLimit: 12,
        maxRotation: 0,
      },
    },
  },
};

const colorPalette = [
  "#0F52BA",
  "#007FFF",
  "#3B82F6",
  "#60A5FA",
  "#93C5FD",
  "#BFDBFE",
  "#DBEAFE",
];

const getColorForMetric = (
  metricName: string,
  defaultIndex: number,
): string => {
  const lower = metricName.toLowerCase();
  if (
    lower.includes("balance") ||
    lower.includes("flow") ||
    lower.includes("optimal") ||
    lower.includes("expected")
  )
    return "#0F52BA";
  if (
    lower.includes("expense") ||
    lower.includes("spend") ||
    lower.includes("careless") ||
    lower.includes("projected")
  )
    return "#EF4444";
  if (lower.includes("income")) return "#10B981";
  return colorPalette[defaultIndex % colorPalette.length];
};

const getProgressiveAnimation = (dataLength: number) => {
  const totalDuration = 800;
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
        if (ctx.type !== "data") {
          return 0;
        }
        if (ctx.index === 0) {
          return ctx.chart.scales.y.getPixelForValue(0);
        }
        const meta = ctx.chart.getDatasetMeta(ctx.datasetIndex);
        const prevElement = meta.data[ctx.index - 1];
        return prevElement
          ? prevElement.getProps(["y"], true).y
          : ctx.chart.scales.y.getPixelForValue(0);
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
    duration: 600,
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
  options?: { disableAnimation?: boolean },
): NativeChartConfig | null => {
  const animationOverride = options?.disableAnimation
    ? { duration: 0, delay: 0 }
    : undefined;

  if (type === "categorized") {
    const allLabels = Array.from(
      new Set(
        payloadData.flatMap((b) =>
          (b.data || []).map((d) => String(d.Category)),
        ),
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
          const pt = (b.data || []).find((d) => String(d.Category) === l);
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
        animation: (animationOverride ||
          getDelayedAnimation()) as unknown as Record<string, unknown>,
        plugins: {
          ...baseOptions.plugins,
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
            borderWidth: 0,
            hoverOffset: 12,
            radius: "80%",
          },
        ],
      },
      options: {
        ...baseOptions,
        cutout: "65%",
        animation: (animationOverride ||
          getDelayedAnimation()) as unknown as Record<string, unknown>,
        scales: { x: { display: false }, y: { display: false } },
        plugins: {
          ...baseOptions.plugins,
        },
      },
    } as unknown as NativeChartConfig;
  }

  if (type === "cash_flow_mixed") {
    const allMonths = Array.from(
      new Set(
        payloadData.flatMap((b) => b.data.map((d) => String(d.Month || ""))),
      ),
    )
      .filter((m) => m !== "")
      .sort((a, b) => {
        try {
          return new Date(`01 ${a}`).getTime() - new Date(`01 ${b}`).getTime();
        } catch {
          return a.localeCompare(b);
        }
      });

    const netBalance = allMonths.map((m) =>
      payloadData.reduce(
        (s, b) =>
          s +
          Number(b.data.find((d) => String(d.Month) === m)?.Net_Balance || 0),
        0,
      ),
    );
    const income = allMonths.map((m) =>
      payloadData.reduce(
        (s, b) =>
          s + Number(b.data.find((d) => String(d.Month) === m)?.Income || 0),
        0,
      ),
    );
    const expense = allMonths.map((m) =>
      payloadData.reduce(
        (s, b) =>
          s + Number(b.data.find((d) => String(d.Month) === m)?.Expense || 0),
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
        animation: (animationOverride ||
          getDelayedAnimation()) as unknown as Record<string, unknown>,
        plugins: {
          ...baseOptions.plugins,
        },
      },
    } as unknown as NativeChartConfig;
  }

  if (type === "health_radar") {
    const dataArray = payloadData[0]?.data || [];
    const labels = dataArray.map((d) => String(d.Metric || ""));
    const scores = dataArray.map((d) => Number(d.Score || 0));

    return {
      type: "radar",
      data: {
        labels,
        datasets: [
          {
            label: "Health Index",
            data: scores,
            backgroundColor: "rgba(0, 127, 255, 0.2)",
            borderColor: "var(--primary)",
            borderWidth: 2,
            pointBackgroundColor: "var(--background)",
            pointBorderColor: "var(--primary)",
            pointHoverBackgroundColor: "#fff",
            pointHoverBorderColor: "var(--primary)",
            pointHoverRadius: 6,
          },
        ],
      },
      options: {
        ...baseOptions,
        animation: (animationOverride ||
          getDelayedAnimation()) as unknown as Record<string, unknown>,
        scales: {
          x: { display: false },
          y: { display: false },
          r: {
            angleLines: { color: "rgba(255, 255, 255, 0.15)" },
            grid: { color: "rgba(255, 255, 255, 0.15)" },
            pointLabels: {
              color: "rgba(255, 255, 255, 0.9)",
              font: { family: "Geist, monospace", size: 11 },
            },
            ticks: { display: false, min: 0, max: 100, stepSize: 20 },
          },
        },
        plugins: {
          ...baseOptions.plugins,
        },
      },
    } as unknown as NativeChartConfig;
  }

  if (type === "expense_forecast" || type === "balance_forecast") {
    const allLabels = Array.from(
      new Set(
        payloadData.flatMap((b) =>
          (b.data || []).map((d) => String(d.Day || d.Month || d.Date || "")),
        ),
      ),
    )
      .filter((l) => l !== "")
      .sort((a, b) => {
        const numA = Number(a.split(" ")[1]);
        const numB = Number(b.split(" ")[1]);
        if (!isNaN(numA) && !isNaN(numB)) return numA - numB;

        try {
          const dateA = new Date(a).getTime();
          const dateB = new Date(b).getTime();
          if (!isNaN(dateA) && !isNaN(dateB)) return dateA - dateB;
        } catch {
          try {
            const mDateA = new Date(`01 ${a}`).getTime();
            const mDateB = new Date(`01 ${b}`).getTime();
            if (!isNaN(mDateA) && !isNaN(mDateB)) return mDateA - mDateB;
          } catch {}
        }
        return a.localeCompare(b);
      });

    const datasets: ChartDataset<"line">[] = [];

    if (type === "balance_forecast") {
      payloadData.forEach((b, i) => {
        const metricColor =
          payloadData.length > 1
            ? colorPalette[i % colorPalette.length]
            : getColorForMetric("balance", i);
        datasets.push({
          label: `${b.bank_name} Expected`,
          data: allLabels.map((label) => {
            const pt = (b.data || []).find(
              (d) => String(d.Day || d.Month || d.Date || "") === label,
            );
            return Number(
              pt?.["Expected Balance"] ||
                pt?.["Balance"] ||
                pt?.["expected_balance"] ||
                pt?.["balance"] ||
                0,
            );
          }),
          borderColor: metricColor,
          backgroundColor: `${metricColor}1A`,
          fill: true,
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
          data: allLabels.map((label) => {
            const pt = (b.data || []).find(
              (d) => String(d.Day || d.Month || d.Date || "") === label,
            );
            return Number(
              pt?.["Projected Daily Spend (£)"] ||
                pt?.["Projected Spend"] ||
                pt?.["spend"] ||
                pt?.["Amount"] ||
                0,
            );
          }),
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
      data: { labels: allLabels, datasets },
      options: {
        ...baseOptions,
        animation: (animationOverride ||
          getProgressiveAnimation(allLabels.length)) as unknown as Record<
          string,
          unknown
        >,
        plugins: {
          ...baseOptions.plugins,
        },
      },
    } as unknown as NativeChartConfig;
  }

  if (type.startsWith("historical")) {
    const allDates = Array.from(
      new Set(
        payloadData.flatMap((b) =>
          (b.data || []).map((d) => String(d.Date || d.Month || d.date || d.month || "")),
        ),
      ),
    )
      .filter((d) => d !== "")
      .sort((a, b) => {
        if (a.includes(" ") || a.includes("-")) {
          try {
            return (
              new Date(`01 ${a}`).getTime() - new Date(`01 ${b}`).getTime()
            );
          } catch {
            return a.localeCompare(b);
          }
        }
        return a.localeCompare(b);
      });

    const datasets: ChartDataset<"line">[] = [];

    payloadData.forEach((b, idx) => {
      const metricColor =
        payloadData.length > 1
          ? colorPalette[idx % colorPalette.length]
          : getColorForMetric("expense", idx);

      datasets.push({
        label: b.bank_name,
        data: allDates.map((date) => {
          const pt = (b.data || []).find(
            (d) => String(d.Date || d.Month || d.date || d.month || "") === date,
          );
          return pt ? Number(pt.Amount || pt.Total_Amount || pt.amount || pt.total_amount || 0) : 0;
        }),
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
        animation: (animationOverride ||
          getProgressiveAnimation(allDates.length)) as unknown as Record<
          string,
          unknown
        >,
        plugins: {
          ...baseOptions.plugins,
        },
      },
    } as unknown as NativeChartConfig;
  }

  return null;
};
