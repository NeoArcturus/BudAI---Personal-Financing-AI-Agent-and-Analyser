"use client";

import React from "react";
import { Card, Skeleton, CloseButton, ProgressBar } from "@heroui/react";
import { getUserUuid } from "@/lib/api";
import { WidgetContext } from "../../../home/DashboardClient";
import WidgetFlipCard from "../../internal/FlipCard";
import { useSubscriptionsData } from "../../../_hooks/useSubscriptionsData";

export function Subscriptions() {
  const { onRemove } = React.useContext(WidgetContext);

  // Using SWR hook for autonomous data fetching and polling
  const uuid = getUserUuid();
  const { subscriptionsData, isLoading, isError, isLocked } = useSubscriptionsData(uuid);

  const formatCurrency = (val: number) =>
    new Intl.NumberFormat("en-GB", {
      style: "currency",
      currency: "GBP",
      maximumFractionDigits: 2,
    }).format(val);

  if (isLocked) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={true} onDiscuss={() => { }}>
        <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden border border-primary/20 bg-primary/5">
          <div className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
            <div className="flex justify-between items-start w-full">
              <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0 animate-pulse">
                [ ANALYSIS SYNCING ]
              </h3>
              <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
            </div>
          </div>
          <div className="flex-1 w-full flex items-center justify-center p-8 pt-0 relative overflow-hidden">
            <div className="w-full h-full rounded-2xl bg-primary/10 animate-pulse border border-primary/20 flex items-center justify-center">
              <span className="text-[10px] font-mono text-primary/60 uppercase tracking-widest">Processing Latest Data...</span>
            </div>
          </div>
        </Card>
      </WidgetFlipCard>
    );
  }

  if (isLoading) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={true} onDiscuss={() => { }}>
        <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
          <div className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
            <div className="flex justify-between items-start w-full">
              <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                Active Subscriptions
              </h3>
              <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
            </div>
          </div>
          <div className="flex-1 w-full flex items-center justify-center p-8 pt-0 relative overflow-hidden">
            <Skeleton className="w-full h-full rounded-2xl bg-white/5" />
          </div>
        </Card>
      </WidgetFlipCard>
    );
  }

  if (isError || !subscriptionsData) {
    return (
      <WidgetFlipCard insight={undefined} isLoading={false} onDiscuss={() => { }}>
        <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
          <div className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
            <div className="flex justify-between items-start w-full">
              <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
                Active Subscriptions
              </h3>
              <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
            </div>
          </div>
          <div className="flex-1 w-full flex flex-col items-center justify-center p-8 pt-0 relative overflow-hidden text-center">
            <p className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">Service Unavailable</p>
          </div>
        </Card>
      </WidgetFlipCard>
    );
  }

  const parseFrequencyToDays = (freq: string | number) => {
    if (typeof freq === "number") return freq;
    const f = String(freq).toLowerCase();
    if (f.includes("month")) return 30.4;
    if (f.includes("week")) return 7;
    if (f.includes("year") || f.includes("annual")) return 365.25;
    if (f.includes("day")) return 1;
    return 30.4;
  };

  return (
    <WidgetFlipCard insight={undefined} isLoading={false} onDiscuss={() => { }}>
      <Card className="w-full h-full liquid-glass rounded-xl flex flex-col relative overflow-hidden">
        <div className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
          <div className="flex justify-between items-start w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0">
              Active Subscriptions
            </h3>
            <CloseButton onPress={onRemove} className="opacity-50 hover:opacity-100 hover:bg-white/10 transition-all rounded-full" />
          </div>

          {subscriptionsData && subscriptionsData.subscriptions.length > 0 && (
            <div className="w-full p-4 border-[0.5px] border-primary/30 bg-primary/5 rounded-xl flex justify-between items-center shadow-inner">
              <span className="text-[9px] font-mono uppercase tracking-[0.3em] text-foreground/60">
                Aggregate Monthly Burden
              </span>
              <span className="text-sm font-black font-mono text-primary tracking-widest">
                {formatCurrency(
                  subscriptionsData.subscriptions.reduce(
                    (acc, sub) => acc + (sub.expected_amount / parseFrequencyToDays(sub.predicted_frequency)) * 30.4, 0
                  )
                )}
              </span>
            </div>
          )}
        </div>

        <div className="flex-1 w-full flex flex-col p-8 pt-0 relative overflow-y-auto scrollbar-hide">
          <div className="flex flex-col gap-3">
            <div className="flex justify-between items-center mb-2 px-4">
              <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">
                Merchant
              </span>
              <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">
                Amount
              </span>
            </div>

            {subscriptionsData.subscriptions.length === 0 ? (
              <div className="p-6 border border-white/5 bg-white/[0.02] rounded-xl flex flex-col items-center justify-center text-center">
                <span className="text-[10px] font-mono uppercase tracking-[0.2em] text-foreground/40">
                  No Subscriptions Found
                </span>
              </div>
            ) : (
              [...subscriptionsData.subscriptions]
                .sort((a, b) => b.expected_amount - a.expected_amount)
                .map((sub, idx) => {
                  const freqDays = parseFrequencyToDays(sub.predicted_frequency);
                  const annualizedCost = (sub.expected_amount / freqDays) * 365.25;

                  let daysRemaining = null;
                  let isDueSoon = false;
                  let progress = 0;

                  if (sub.next_expected_date) {
                    const nextDate = new Date(sub.next_expected_date);
                    const today = new Date();
                    // Strip time for accurate day count
                    today.setHours(0, 0, 0, 0);
                    nextDate.setHours(0, 0, 0, 0);
                    const diffTime = nextDate.getTime() - today.getTime();
                    daysRemaining = Math.ceil(diffTime / (1000 * 60 * 60 * 24));
                    isDueSoon = daysRemaining <= 1;

                    if (sub.last_payment_date) {
                      const lastDate = new Date(sub.last_payment_date);
                      lastDate.setHours(0, 0, 0, 0);
                      const totalDuration = nextDate.getTime() - lastDate.getTime();
                      const elapsed = today.getTime() - lastDate.getTime();
                      if (totalDuration > 0) {
                        progress = Math.min(100, Math.max(0, (elapsed / totalDuration) * 100));
                      }
                    }
                  }

                  let priceIncrease = 0;
                  if (sub.is_price_hike && sub.last_payment_amount) {
                    priceIncrease = sub.expected_amount - sub.last_payment_amount;
                  }

                  const maskedAcc = sub.account_number ? `••••${sub.account_number.slice(-4)}` : "";
                  const bankDetails = sub.sort_code && maskedAcc
                    ? `(${sub.sort_code}, ${maskedAcc})`
                    : maskedAcc ? `(${maskedAcc})` : "";

                  console.log(sub.account_number, sub.bank_name, sub.sort_code)

                  return (
                    <div key={idx} className="flex flex-col gap-3 p-4 rounded-xl bg-white/[0.02] border border-white/5 hover:border-primary/20 transition-colors">
                      <div className="flex justify-between items-start">
                        <div className="flex flex-col">
                          <span className="text-foreground font-bold tracking-wide text-sm uppercase">
                            {sub.merchant_name}
                          </span>
                          <span className="text-[10px] font-mono tracking-[0.2em] text-foreground/40 mt-0.5">
                            Account: {sub.bank_name || "Unknown"} {bankDetails}
                          </span>
                        </div>
                        <div className="flex flex-col items-end gap-1 text-right">
                          <span className="text-foreground font-mono font-black text-sm">
                            {formatCurrency(sub.expected_amount)}
                          </span>
                          <span className="text-[9px] font-mono text-foreground/40 uppercase tracking-widest leading-tight">
                            Yearly Cost: {formatCurrency(annualizedCost)}
                          </span>
                        </div>
                      </div>

                      <div className="flex justify-between items-end mt-2">
                        <span className="text-[10px] font-mono tracking-[0.2em] text-foreground/40">
                          Billed: {String(sub.predicted_frequency).toUpperCase()}
                        </span>
                        {daysRemaining !== null && (
                          <span className={`text-[10px] font-mono tracking-[0.2em] ${isDueSoon ? 'text-danger font-bold animate-pulse' : 'text-primary'}`}>
                            Due in: {daysRemaining} Day{daysRemaining !== 1 ? 's' : ''}
                          </span>
                        )}
                      </div>

                      <ProgressBar
                        aria-label="Billing cycle progress"
                        value={progress}
                        size="sm"
                        className="w-full mt-2"
                      >
                        <ProgressBar.Track className="bg-white/5">
                          <ProgressBar.Fill className="bg-primary/40" />
                        </ProgressBar.Track>
                      </ProgressBar>

                      {sub.is_price_hike && priceIncrease > 0 && (
                        <div className="mt-1 pt-2 border-t border-danger/20 flex justify-between items-center">
                          <span className="text-danger font-mono text-[9px] uppercase tracking-widest">
                            Price Increase: +{formatCurrency(priceIncrease)} (was {formatCurrency(sub.last_payment_amount!)})
                          </span>
                        </div>
                      )}
                    </div>
                  );
                })
            )}
          </div>
        </div>
      </Card>
    </WidgetFlipCard>
  );
}
