import { cookies } from "next/headers";
import LedgerTableWidgetClient from "./LedgerTableWidgetClient";
import { Transaction } from "@/types";

export default async function LedgerTableWidget() {
  const cookieStore = await cookies();
  const token = cookieStore.get("budai_token")?.value;
  const baseUrl =
    process.env.NEXT_PUBLIC_API_BASE_URL || "http://localhost:8080";

  let initialData: Transaction[] | undefined = undefined;

  if (token) {
    try {
      const accountsRes = await fetch(`${baseUrl}/api/accounts`, {
        headers: { Authorization: `Bearer ${token}` },
      });
      if (accountsRes.ok) {
        const accountsData = await accountsRes.json();
        const accountIds = (accountsData.accounts || []).map((a: any) => a.account_id);

        if (accountIds.length > 0) {
          const fromDate = "2025-01-01";
          const toDate = new Date().toISOString().split("T")[0];

          const res = await fetch(
            `${baseUrl}/api/accounts/${accountIds.join(",")}/transactions?from=${fromDate}&to=${toDate}`,
            {
              headers: { Authorization: `Bearer ${token}` },
            },
          );
          if (res.ok) {
            const result = await res.json();
            if (result.transactions && result.transactions.length > 0) {
              initialData = result.transactions;
            }
          }
        }
      }
    } catch (e) {
      console.error("LedgerTableWidget pre-fetch failed:", e);
    }
  }

  return <LedgerTableWidgetClient initialData={initialData} />;
}
