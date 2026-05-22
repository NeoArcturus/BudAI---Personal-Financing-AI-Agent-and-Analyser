import { cookies } from "next/headers";
import SpendingTrendWidgetClient from "./SpendingTrendWidgetClient";
import { BankChartData } from "@/types";

export default async function SpendingTrendWidget() {
  const cookieStore = await cookies();
  const token = cookieStore.get("budai_token")?.value;
  const baseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

  let initialData: BankChartData[] | undefined = undefined;

  if (token) {
    try {
      const accountsRes = await fetch(`${baseUrl}/api/accounts`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (accountsRes.ok) {
        const accountsData = await accountsRes.json();
        const accountIds = (accountsData.accounts || []).map((a: any) => a.account_id);

        if (accountIds.length > 0) {
          const toDate = new Date().toISOString().split("T")[0];
          const fromDate = new Date(Date.now() - 180 * 24 * 60 * 60 * 1000)
            .toISOString()
            .split("T")[0];

          const res = await fetch(`${baseUrl}/api/media/execute`, {
            method: "POST",
            headers: {
              "Content-Type": "application/json",
              Authorization: `Bearer ${token}`,
            },
            body: JSON.stringify({
              tool_name: "plot_expenses",
              parameters: {
                plot_time_type: "monthly",
                from_date: fromDate,
                to_date: toDate,
                bank_name_or_id: accountIds.join(","),
              },
            }),
          });

          if (res.ok) {
            const result = await res.json();
            if (result.data && result.data.length > 0) {
              initialData = result.data;
            }
          }
        }
      }
    } catch (e) {
      console.error("SpendingTrendWidget pre-fetch failed:", e);
    }
  }

  return <SpendingTrendWidgetClient initialData={initialData} />;
}
