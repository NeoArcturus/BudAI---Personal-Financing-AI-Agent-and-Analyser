import LedgerTableWidget from "@/app/(protected)/_components/LedgerTableWidget";
import TransactionsClient from "./TransactionsClient";

export default async function TransactionsPage() {
  return (
    <TransactionsClient
      ledgerTable={<LedgerTableWidget />}
    />
  );
}
