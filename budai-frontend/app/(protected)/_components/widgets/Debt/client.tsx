"use client";
import BucketListWidget from "../BucketLists/client";
import { CreditCard } from "lucide-react";
export default function DebtWidget() {
  return <BucketListWidget bucketType="LIABILITY" title="Liabilities" icon={CreditCard} />;
}
