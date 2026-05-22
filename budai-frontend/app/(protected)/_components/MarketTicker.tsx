import { cookies } from "next/headers";
import MarketTickerClient from "./MarketTickerClient";

export default async function MarketTicker() {
  const cookieStore = await cookies();
  const token = cookieStore.get("budai_token")?.value;
  const baseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

  let initialData = undefined;

  if (token) {
    try {
      const res = await fetch(`${baseUrl}/api/market/ticker`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (res.ok) {
        const json = await res.json();
        if (json.tickers && json.tickers.length > 0) {
          initialData = json.tickers;
        }
      }
    } catch (e) {
      console.error("MarketTicker pre-fetch failed:", e);
    }
  }

  return <MarketTickerClient initialData={initialData} />;
}
