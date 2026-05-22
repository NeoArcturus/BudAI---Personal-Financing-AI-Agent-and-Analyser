import { cookies } from "next/headers";
import CommodityMarketWidgetClient from "./CommodityMarketWidgetClient";

export default async function CommodityMarketWidget() {
  const cookieStore = await cookies();
  const token = cookieStore.get("budai_token")?.value;
  const baseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

  let initialHistory = undefined;

  if (token) {
    try {
      const historyRes = await fetch(`${baseUrl}/api/market/history`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (historyRes.ok) {
        const historyData = await historyRes.json();
        if (historyData.history && historyData.history.length > 0) {
          initialHistory = historyData;
        }
      }
    } catch (e) {
      console.error("CommodityMarketWidget pre-fetch failed:", e);
    }
  }

  return (
    <CommodityMarketWidgetClient
      initialHistory={initialHistory}
    />
  );
}
