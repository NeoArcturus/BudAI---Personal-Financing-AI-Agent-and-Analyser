import { NativeChartConfig, ToolParameters, BankChartData } from "@/types";
import { ChartDataset } from "chart.js";

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

const colorPalette = [
  "#00FFAA",
  "#3b82f6",
  "#ef4444",
  "#f59e0b",
  "#a855f7",
  "#ec4899",
];

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
    const datasets = payloadData.map((b, i) => ({
      label: `${b.bank_name} Spent (£)`,
      data: allLabels.map((l) => {
        const pt = b.data.find((d) => String(d.Category) === l);
        return pt ? Number(pt.Total_Amount) : 0;
      }),
      backgroundColor: colorPalette[i % colorPalette.length],
      borderRadius: 6,
      hoverBackgroundColor: `${colorPalette[i % colorPalette.length]}cc`,
    }));

    return {
      type: "bar",
      data: { labels: allLabels, datasets },
      options: {
        ...baseOptions,
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
            borderColor: "#3b82f6",
            backgroundColor: "rgba(59, 130, 246, 0.1)",
            borderWidth: 3,
            tension: 0.4,
            fill: true,
            pointBackgroundColor: "#3b82f6",
            pointRadius: 3,
            pointHoverRadius: 6,
          },
          {
            type: "bar",
            label: "Income (£)",
            data: income,
            backgroundColor: "#00FFAA",
            borderRadius: 4,
          },
          {
            type: "bar",
            label: "Expenses (£)",
            data: expense,
            backgroundColor: "#ef4444",
            borderRadius: 4,
          },
        ],
      },
      options: {
        ...baseOptions,
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
        datasets.push({
          label: `${b.bank_name} Expected`,
          data: allDays.map((day) =>
            Number(
              b.data.find((d) => d.Day === day)?.["Expected Balance"] || 0,
            ),
          ),
          borderColor: colorPalette[i % colorPalette.length],
          backgroundColor: `${colorPalette[i % colorPalette.length]}1A`,
          fill: false,
          tension: 0.4,
          pointRadius: 0,
          pointHitRadius: 10,
          pointHoverRadius: 5,
        });
      });
    } else {
      payloadData.forEach((b, i) => {
        datasets.push({
          label: `${b.bank_name} Spend`,
          data: allDays.map((day) =>
            Number(
              b.data.find((d) => d.Day === day)?.[
                "Projected Daily Spend (£)"
              ] || 0,
            ),
          ),
          borderColor: colorPalette[i % colorPalette.length],
          backgroundColor: `${colorPalette[i % colorPalette.length]}1A`,
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
      datasets.push({
        label: b.bank_name,
        data: allDates.map((date) =>
          Number(b.data.find((d) => d.Date === date)?.Amount || 0),
        ),
        borderColor: colorPalette[idx % colorPalette.length],
        fill: false,
        tension: 0.4,
        pointRadius: 2,
        pointBackgroundColor: colorPalette[idx % colorPalette.length],
        pointHitRadius: 10,
        pointHoverRadius: 6,
      });
    });

    return {
      type: "line",
      data: { labels: allDates, datasets },
      options: {
        ...baseOptions,
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
