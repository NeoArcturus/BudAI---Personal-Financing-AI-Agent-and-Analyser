"use client";
import BucketListWidget from "../BucketLists/client";
import { Target } from "lucide-react";
export default function GoalsWidget() {
  return <BucketListWidget bucketType="TARGET_DATE" title="Goals Progress" icon={Target} />;
}
