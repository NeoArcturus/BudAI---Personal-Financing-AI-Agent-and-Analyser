import { cookies } from "next/headers";
import FinancialNewsWidgetClient from "./FinancialNewsWidgetClient";

export default async function FinancialNewsWidget() {
  const cookieStore = await cookies();
  const token = cookieStore.get("budai_token")?.value;
  const baseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

  let initialNews = undefined;

  if (token) {
    try {
      const newsRes = await fetch(`${baseUrl}/api/market/news`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (newsRes.ok) {
        const newsData = await newsRes.json();
        if ((newsData.geopolitical && newsData.geopolitical.length > 0) || (newsData.market && newsData.market.length > 0)) {
          initialNews = newsData;
        }
      }
    } catch (e) {
      console.error("FinancialNewsWidget pre-fetch failed:", e);
    }
  }

  return (
    <FinancialNewsWidgetClient
      initialNews={initialNews}
    />
  );
}
