"use client";
import BucketListWidget from "../BucketLists/client";
import { Repeat } from "lucide-react";
export default function RecurringWidget() {
  return <BucketListWidget bucketType="RECURRING" title="Recurring Payments" icon={Repeat} />;
}
