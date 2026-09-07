"use client";

import React, { useState } from "react";
import { useQuery, useMutation, useQueryClient } from "@tanstack/react-query";
import { usePersistedState } from "@/lib/hooks";
import { 
  Card, 
  Tabs, 
  Tab, 
  Avatar, 
  ProgressBar, Meter,
  Chip, 
  Modal, 
  Button,
  CloseButton,
  Label
} from "@heroui/react";
import { Target, Repeat, CreditCard, ExternalLink, ArrowRightCircle } from "lucide-react";
import { apiFetch } from "@/lib/api";
import WidgetFlipCard, { FlipButton } from "../../internal/FlipCard";
import { WidgetContext } from "../../../home/DashboardClient";
import { AnimatedNumber } from "../../ui/AnimatedNumber";

interface BucketListProps {
  bucketType: "TARGET_DATE" | "RECURRING" | "LIABILITY";
  title: string;
  icon: any;
}

export default function BucketListWidget({ bucketType, title, icon: Icon }: BucketListProps) {
  const queryClient = useQueryClient();
  const { onRemove } = React.useContext(WidgetContext);
  const [selectedInactiveBucket, setSelectedInactiveBucket] = useState<any | null>(null);
  const { instanceId } = React.useContext(WidgetContext);
  const [activeTab, setActiveTab] = usePersistedState<"active" | "archived">(`bucket_tab_${instanceId || bucketType}`, "active");

  // Use the exact same query key as DashboardClient to share cache
  const { data: buckets = [] } = useQuery({
    queryKey: ["buckets"],
    queryFn: () => apiFetch("/api/buckets"),
    staleTime: 1000 * 60,
  });

  const reactivateMutation = useMutation({
    mutationFn: async (id: string) => {
      return apiFetch(`/api/buckets/${id}/active`, {
        method: "PATCH",
        body: JSON.stringify({ is_active: true }),
      });
    },
    onSuccess: () => {
      queryClient.invalidateQueries({ queryKey: ["buckets"] });
      setSelectedInactiveBucket(null);
    },
  });

  // Filter buckets for this specific widget type
  const safeBuckets = Array.isArray(buckets) ? buckets : (buckets as any).data || [];
  const typeBuckets = safeBuckets.filter((b: any) => b.type === bucketType);
  const activeBuckets = typeBuckets.filter((b: any) => b.is_active !== false);
  const inactiveBuckets = typeBuckets.filter((b: any) => b.is_active === false);

  const handleActiveClick = (id: string) => {
    const el = document.getElementById(id);
    if (el) {
      el.scrollIntoView({ behavior: "smooth", block: "center" });
      
      // Add a temporary glow effect
      const inner = el.querySelector(".group") as HTMLElement;
      if (inner) {
        inner.style.transition = "box-shadow 0.3s ease";
        inner.style.boxShadow = "0 0 30px rgba(0, 242, 255, 0.6)";
        setTimeout(() => {
          inner.style.boxShadow = "none";
        }, 1500);
      }
    }
  };

  const calculateTotal = (list: any[]) => {
    return list.reduce((sum, b) => sum + (b.cached_balance || 0), 0);
  };

  const activeTotal = calculateTotal(activeBuckets);

  return (
    <WidgetFlipCard>
      <Card className="w-full h-full">
        <Card.Header className="flex flex-col gap-6 p-8 shrink-0 w-full z-10">
          <div className="flex justify-between items-start w-full">
            <h3 className="text-[10px] font-black text-primary uppercase tracking-[0.4em] italic m-0 flex items-center gap-2">
              <Icon size={12} className="text-primary" />
              {title}
            </h3>
            <div className="flex items-start gap-4">
              <p className="text-[9px] text-foreground/40 font-black uppercase tracking-[0.3em] m-0 text-right mr-2">
                Total<br/>
                <span className="text-foreground text-[10px] tracking-widest"><AnimatedNumber value={activeTotal} minimumFractionDigits={0} maximumFractionDigits={0} /></span>
              </p>
              <div className="flex items-center gap-1 -mt-2 -mr-2">
                <FlipButton />
                <CloseButton
                  onPress={onRemove}
                  className="w-8 h-8 min-w-8 opacity-50 hover:opacity-100 hover:bg-white/10 text-foreground transition-all rounded-full"
                />
              </div>
            </div>
          </div>
        </Card.Header>

        <Card.Content className="flex-1 w-full flex flex-col p-8 pt-0 relative overflow-hidden pointer-events-auto">
          <div className="flex items-center gap-4 mb-4 border-b border-white/10 pb-2">
            <button 
              onClick={() => setActiveTab("active")}
              className={`text-xs font-bold uppercase tracking-widest pb-2 border-b-2 transition-all ${activeTab === "active" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-white"}`}>
              Active ({activeBuckets.length})
            </button>
            <button 
              onClick={() => setActiveTab("archived")}
              className={`text-xs font-bold uppercase tracking-widest pb-2 border-b-2 transition-all ${activeTab === "archived" ? "border-primary text-primary" : "border-transparent text-muted-foreground hover:text-white"}`}>
              Archived ({inactiveBuckets.length})
            </button>
          </div>

          <div className="w-full flex-1 overflow-y-auto scrollbar-hide flex flex-col gap-2 pb-4">
            {activeTab === "active" ? (
              activeBuckets.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground/40 h-32">
                  <span className="text-xs uppercase tracking-widest font-black">No Active {title}</span>
                </div>
              ) : (
                activeBuckets.map((bucket: any) => (
                  <button
                    key={bucket.id}
                    onClick={() => handleActiveClick(bucket.id)}
                    className="w-full flex flex-col gap-2 bg-white/5 hover:bg-white/10 border-[0.5px] border-white/5 hover:border-primary/30 p-4 rounded-2xl transition-all cursor-pointer group/row outline-none text-left"
                  >
                    <div className="w-full flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="bg-primary/10 text-primary border border-primary/20 w-8 h-8 rounded-full  shrink-0 flex items-center justify-center">
                          <Icon size={14} />
                        </div>
                        <span className="font-bold text-sm tracking-wide text-foreground group-hover/row:text-primary transition-colors truncate">
                          {bucket.name || bucket.title}
                        </span>
                      </div>
                      <span className="font-mono text-sm tracking-tighter text-foreground">
                        <AnimatedNumber value={bucket.cached_balance || 0} />
                      </span>
                    </div>
                    {bucketType === "TARGET_DATE" && bucket.target_amount && (
                      <Meter
                        aria-label="Progress"
                        value={Math.min(100, Math.max(0, ((bucket.cached_balance || 0) / bucket.target_amount) * 100))}
                        className="w-full mt-2 block"
                      >
                        <div className="flex justify-between text-[9px] uppercase tracking-widest font-black text-foreground/40 mb-1 w-full">
                          <Label>Progress</Label>
                          <Meter.Output />
                        </div>
                        <Meter.Track className="bg-white/10 h-1.5 w-full rounded-full overflow-hidden">
                          <Meter.Fill className="bg-primary rounded-full" />
                        </Meter.Track>
                      </Meter>
                    )}
                  </button>
                ))
              )
            ) : (
              inactiveBuckets.length === 0 ? (
                <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground/40 h-32">
                  <span className="text-xs uppercase tracking-widest font-black">No Archived {title}</span>
                </div>
              ) : (
                inactiveBuckets.map((bucket: any) => (
                  <button
                    key={bucket.id}
                    onClick={() => setSelectedInactiveBucket(bucket)}
                    className="w-full flex flex-col gap-2 bg-white/5 hover:bg-white/10 border-[0.5px] border-white/5 hover:border-white/20 p-4 rounded-2xl transition-all cursor-pointer group/row outline-none text-left opacity-70 hover:opacity-100"
                  >
                    <div className="w-full flex items-center justify-between">
                      <div className="flex items-center gap-3">
                        <div className="w-8 h-8 rounded-full bg-white/5 text-muted-foreground border border-white/10 shrink-0 flex items-center justify-center">
                          <Icon size={14} />
                        </div>
                        <span className="font-bold text-sm tracking-wide text-foreground truncate flex items-center gap-2">
                          {bucket.name || bucket.title}
                          <Chip size="sm" variant="soft" className="scale-75 origin-left">Archived</Chip>
                        </span>
                      </div>
                      <span className="font-mono text-sm tracking-tighter text-foreground/70">
                        <AnimatedNumber value={bucket.cached_balance || 0} />
                      </span>
                    </div>
                    {bucketType === "TARGET_DATE" && bucket.target_amount && (
                      <Meter
                        aria-label="Progress"
                        value={Math.min(100, Math.max(0, ((bucket.cached_balance || 0) / bucket.target_amount) * 100))}
                        className="w-full mt-2 block"
                      >
                        <div className="flex justify-between text-[9px] uppercase tracking-widest font-black text-foreground/40 mb-1 w-full">
                          <Label>Progress</Label>
                          <Meter.Output />
                        </div>
                        <Meter.Track className="bg-white/10 h-1.5 w-full rounded-full overflow-hidden">
                          <Meter.Fill className={bucket.cached_balance >= bucket.target_amount ? "bg-green-500 rounded-full" : "bg-primary rounded-full"} />
                        </Meter.Track>
                      </Meter>
                    )}
                  </button>
                ))
              )
            )}
          </div>
        </Card.Content>

        <Modal.Backdrop isOpen={!!selectedInactiveBucket} onOpenChange={(open) => !open && setSelectedInactiveBucket(null)} variant="blur">
          <Modal.Container>
            <Modal.Dialog className="modal p-6 overflow-hidden max-w-sm w-full">
              <div className="absolute inset-0 bg-[radial-gradient(ellipse_at_top,rgba(255,255,255,0.05)_0%,transparent_70%)] pointer-events-none" />
              <Modal.Header className="flex items-center justify-between z-10 relative">
                <Modal.Heading className="text-xl font-bold tracking-tight text-foreground">
                  Archived Asset
                </Modal.Heading>
                <Modal.CloseTrigger className="text-muted-foreground hover:text-foreground transition-colors cursor-pointer" />
              </Modal.Header>
              <Modal.Body className="py-6 z-10 relative">
                {selectedInactiveBucket && (
                  <div className="flex flex-col gap-4">
                    <div className="flex items-center gap-4 p-4 bg-white/5 rounded-xl border border-white/5">
                      <div className="w-12 h-12 rounded-xl bg-white/5 flex items-center justify-center text-muted-foreground shrink-0">
                        <Icon size={24} />
                      </div>
                      <div>
                        <h3 className="font-bold text-base tracking-wide">{selectedInactiveBucket.name || selectedInactiveBucket.title}</h3>
                        <p className="text-xs text-muted-foreground uppercase tracking-widest font-mono">
                          {selectedInactiveBucket.type}
                        </p>
                      </div>
                    </div>
                    <div className="flex flex-col gap-1 px-1">
                      <Label className="text-[10px] font-black uppercase tracking-widest text-foreground/40">Final Balance</Label>
                      <span className="font-mono text-2xl tracking-tighter text-foreground">
                        <AnimatedNumber value={selectedInactiveBucket.cached_balance || 0} />
                      </span>
                    </div>
                  </div>
                )}
              </Modal.Body>
              <Modal.Footer className="flex justify-between items-center mt-4 border-t border-white/10 pt-4 z-10 relative">
                <Button variant="ghost" onPress={() => setSelectedInactiveBucket(null)} className="text-muted-foreground cursor-pointer hover:bg-white/5">
                  Cancel
                </Button>
                <Button 
                  variant="primary" 
                  isDisabled={reactivateMutation.isPending}
                  onPress={() => selectedInactiveBucket && reactivateMutation.mutate(selectedInactiveBucket.id)}
                  className="bg-primary text-primary-foreground font-bold tracking-wide shadow-[0_0_15px_rgba(0,242,255,0.3)] hover:shadow-[0_0_25px_rgba(0,242,255,0.5)] transition-all cursor-pointer"
                >
                  {reactivateMutation.isPending ? "Reactivating..." : "Reactivate Asset"}
                </Button>
              </Modal.Footer>
            </Modal.Dialog>
          </Modal.Container>
        </Modal.Backdrop>
      </Card>
    </WidgetFlipCard>
  );
}
