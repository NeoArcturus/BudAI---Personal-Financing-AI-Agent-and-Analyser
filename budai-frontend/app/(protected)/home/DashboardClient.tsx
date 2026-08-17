"use client";

import React, { useState, useEffect } from "react";
import { useRouter } from "next/navigation";
import { Button, CloseButton, SearchField } from "@heroui/react";
import {
  Globe,
  Bell,
  Clock,
  CreditCard,
  Moon,
  Sun,
  Plus,
  LineChart,
  BarChart,
  PieChart,
  MessageSquare,
  Sparkles,
  TrendingUp,
  Activity,
  Repeat,
  AlertTriangle,
  ShieldAlert,
  Heart,
  LayoutDashboard
} from "lucide-react";
import { useBudAI } from "@/app/context/AppContext";
import { useTheme } from "next-themes";
import { NLPDashboardController } from "@/components/dashboard/NLPDashboardController";
import { ProactiveInsightsFeed } from "@/components/dashboard/ProactiveInsightsFeed";
import { useQuery } from "@tanstack/react-query";
import { apiFetch } from "@/lib/api";
import { useUserProfile } from "@/lib/hooks";

import {
  DndContext,
  closestCenter,
  KeyboardSensor,
  PointerSensor,
  useSensor,
  useSensors,
  DragEndEvent,
} from "@dnd-kit/core";
import {
  arrayMove,
  SortableContext,
  sortableKeyboardCoordinates,
  rectSortingStrategy,
  useSortable,
} from "@dnd-kit/sortable";
import { CSS } from "@dnd-kit/utilities";

interface DashboardClientProps {
  widgetsMap?: Record<string, React.ReactNode>;
  ticker?: React.ReactNode;
}

export const WidgetContext = React.createContext<{ onRemove?: () => void; instanceId?: string; }>({});

type LocalWidgetState = {
  widget_uuid: string;
  type: string;
  title: string;
  localHeight: number;
  localColSpan: number;
  isManual?: boolean;
};

const AVAILABLE_WIDGET_TYPES = [
  { type: "cashFlow", label: "Cash Flow - Income vs Expense", icon: BarChart },
  { type: "spendingTrend", label: "Spending Trend - Historical", icon: LineChart },
  { type: "expenseDistribution", label: "Expense Distribution", icon: PieChart },
  { type: "ledger", label: "Ledger Table - Historical Transactions", icon: Clock },

  { type: "commodityMarket", label: "Market Intelligence - Commodities", icon: Globe },
  { type: "financialNews", label: "Global Financial Feed - News", icon: MessageSquare },
  { type: "balanceForecast", label: "Account Balance Forecast", icon: TrendingUp },
  { type: "aiChat", label: "AI Chat Session Headings", icon: MessageSquare },
  { type: "analyticsHabits", label: "Analytics - Spending Habits", icon: Activity },
  { type: "analyticsSubscriptions", label: "Analytics - Subscriptions", icon: Repeat },
  { type: "analyticsAnomalies", label: "Analytics - Anomalies", icon: AlertTriangle },
  { type: "analyticsRisk", label: "Analytics - Risk Assessment", icon: ShieldAlert },

  { type: "analyticsHealth", label: "Analytics - Financial Health", icon: Heart },
];

function SortableWidgetItem({
  id,
  height,
  colSpan,
  onResize,
  children,
}: {
  id: string;
  height: number;
  colSpan: number;
  onResize: (id: string, newHeight: number, newColSpan: number) => void;
  children: React.ReactNode;
}) {
  const { attributes, listeners, setNodeRef, transform, transition, isDragging } =
    useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    height: `${height}px`,
    gridColumn: `span ${colSpan} / span ${colSpan}`,
    zIndex: isDragging ? 50 : 1,
  };

  const handleResizePointerDown = (e: React.PointerEvent<HTMLDivElement>) => {
    e.stopPropagation();
    e.preventDefault();

    const startY = e.clientY;
    const startX = e.clientX;
    const startHeight = height;
    const startColSpan = colSpan;

    const onPointerMove = (moveEvent: PointerEvent) => {
      const deltaY = moveEvent.clientY - startY;
      const deltaX = moveEvent.clientX - startX;

      const newHeight = Math.max(300, startHeight + deltaY);

      let newColSpan = startColSpan;
      if (deltaX > 100) {
        newColSpan = 2;
      } else if (deltaX < -100) {
        newColSpan = 1;
      }

      onResize(id, newHeight, newColSpan);
    };

    const onPointerUp = () => {
      document.removeEventListener("pointermove", onPointerMove);
      document.removeEventListener("pointerup", onPointerUp);
    };

    document.addEventListener("pointermove", onPointerMove);
    document.addEventListener("pointerup", onPointerUp);
  };

  return (
    <div
      ref={setNodeRef}
      style={style}
      className="relative flex touch-none group w-full bg-content2 border border-white/5 shadow-2xl rounded-[32px] overflow-hidden"
    >
      <div
        className="w-full h-full cursor-grab active:cursor-grabbing"
        {...attributes}
        {...listeners}
      >
        {children}
      </div>

      <div
        onPointerDown={handleResizePointerDown}
        className="absolute bottom-0 right-0 w-6 h-6 cursor-nwse-resize opacity-0 group-hover:opacity-100 transition-opacity z-50 flex items-center justify-center"
      >
        <div className="w-3 h-3 border-r-2 border-b-2 border-primary/30 rounded-br-sm" />
      </div>
    </div>
  );
}

export default function DashboardClient({
  ticker,
  widgetsMap = {},
}: DashboardClientProps) {
  const { userName, sessions, accounts } = useBudAI();
  const { theme, setTheme } = useTheme();
  const router = useRouter();
  const [mounted, setMounted] = useState(false);
  const [localWidgets, setLocalWidgets] = useState<LocalWidgetState[]>([]);
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [manualLoaded, setManualLoaded] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  // Load manual widgets from local storage on mount
  useEffect(() => {
    if (!userName || userName === "User") return;
    const storageKey = `budai_manual_widgets_${userName}`;
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      try {
        const parsed = JSON.parse(saved);
        setLocalWidgets((prev) => {
          const aiWidgets = prev.filter(w => !w.isManual);
          return [...aiWidgets, ...parsed];
        });
      } catch (e) {
        console.error("Failed to parse manual widgets:", e);
      }
    } else {
      // Default manual widget for new users if needed, or leave empty
      const defaultWidgets: LocalWidgetState[] = [
        { widget_uuid: `manual-portfolio-1`, type: "portfolio", title: "Portfolio", isManual: true, localHeight: 450, localColSpan: 1 },
      ];
      setLocalWidgets((prev) => {
        const aiWidgets = prev.filter(w => !w.isManual);
        return [...aiWidgets, ...defaultWidgets];
      });
    }
    setManualLoaded(true);
  }, [userName]);

  // Save manual widgets to local storage when they change
  useEffect(() => {
    if (!manualLoaded || !userName || userName === "User") return;
    const manualWidgets = localWidgets.filter(w => w.isManual);
    const storageKey = `budai_manual_widgets_${userName}`;
    localStorage.setItem(storageKey, JSON.stringify(manualWidgets));
  }, [localWidgets, manualLoaded, userName]);

  const { data: widgetsResponse, isLoading, isError } = useQuery({
    queryKey: ["dashboard-widgets"],
    queryFn: async () => {
      const response = await apiFetch("/api/dashboard/widgets", {
        method: "GET",
      }, true);
      if (!response.ok) {
        throw new Error("Failed to fetch widgets");
      }
      return (await response.json()) as { widgets: string[] };
    },
    refetchInterval: 5000, 
  });

  const { data: userProfile, isLoading: isLoadingProfile } = useUserProfile();

  // Synchronization Gate (moved below hooks)

  useEffect(() => {
    if (!widgetsResponse?.widgets) return;

    setLocalWidgets((prev) => {
      const manualWidgets = prev.filter(w => w.isManual);
      const aiWidgets = prev.filter(w => !w.isManual);
      
      const currentIds = new Set(aiWidgets.map((w) => w.type));
      const backendIds = new Set(widgetsResponse.widgets);

      // Remove AI widgets that don't exist in backend anymore
      const filteredAi = aiWidgets.filter((w) => backendIds.has(w.type));

      // Add new AI widgets
      widgetsResponse.widgets.forEach((type) => {
        if (!currentIds.has(type)) {
          filteredAi.push({
            widget_uuid: `ai-${type}-${crypto.randomUUID()}`,
            type: type,
            title: type,
            isManual: false,
            localHeight: 450,
            localColSpan: 1,
          });
        }
      });
      
      // Combine manual and AI
      return [...filteredAi, ...manualWidgets];
    });
  }, [widgetsResponse?.widgets]);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      setLocalWidgets((items) => {
        const oldIndex = items.findIndex((w) => w.widget_uuid === active.id);
        const newIndex = items.findIndex((w) => w.widget_uuid === over.id);
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  };

  const handleResizeWidget = (id: string, newHeight: number, newColSpan: number) => {
    setLocalWidgets((current) =>
      current.map((w) =>
        w.widget_uuid === id
          ? { ...w, localHeight: newHeight, localColSpan: newColSpan }
          : w,
      ),
    );
  };

  const handleRemoveWidget = (idToRemove: string) => {
    setLocalWidgets((current) => current.filter((w) => w.widget_uuid !== idToRemove));
    // Note: If removing an AI widget, it will reappear on next poll unless removed in backend.
    // Ideally we should hit a DELETE endpoint here if it's not manual.
  };

  const handleAddManualWidget = (type: string) => {
    const newWidget: LocalWidgetState = {
      widget_uuid: `manual-${type}-${crypto.randomUUID()}`,
      type,
      title: type,
      isManual: true,
      localHeight: 450,
      localColSpan: 1,
    };
    setLocalWidgets((current) => [...current, newWidget]);
    setIsModalOpen(false);
  };

  const renderWidgetContent = (widget: LocalWidgetState) => {
    const widgetElement = widgetsMap[widget.type];
    if (widgetElement) {
      return (
        <WidgetContext.Provider value={{ onRemove: () => handleRemoveWidget(widget.widget_uuid) }}>
          {widgetElement}
        </WidgetContext.Provider>
      );
    }
    if (widget.type === "aiChat") {
      return (
        <div className="w-full h-full liquid-glass rounded-3xl flex flex-col relative shadow-xl overflow-hidden p-6 gap-4">
          <div className="flex items-center gap-3 text-muted-foreground/40 mb-2">
            <Sparkles size={20} className="text-primary" />
            <span className="text-xs font-bold uppercase tracking-widest text-primary">
              Recent AI Chats
            </span>
          </div>
          <div className="flex flex-col gap-2 overflow-y-auto scrollbar-hide flex-1">
            {sessions.length > 0 ? (
              sessions.slice(0, 5).map((session) => (
                <div
                  key={session.id}
                  onClick={() => router.push(`/advisor?session=${session.id}`)}
                  className="p-3 bg-white/5 hover:bg-white/10 border border-white/10 rounded-xl cursor-pointer transition-colors flex items-center justify-between group"
                >
                  <div className="flex items-center gap-3 overflow-hidden">
                    <MessageSquare size={16} className="text-muted-foreground group-hover:text-primary transition-colors shrink-0" />
                    <span className="text-sm font-medium text-white/80 group-hover:text-white truncate">
                      {session.title || "New Session"}
                    </span>
                  </div>
                </div>
              ))
            ) : (
              <div className="flex-1 flex flex-col items-center justify-center text-muted-foreground/50 gap-2">
                <MessageSquare size={24} />
                <span className="text-xs uppercase tracking-wider">No recent chats</span>
              </div>
            )}
          </div>
          <CloseButton
            onClick={() => handleRemoveWidget(widget.widget_uuid)}
            className="absolute top-4 right-4 text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors rounded-2xl"
          />
        </div>
      );
    }
    return null;
  };

  // Synchronization Gate - Render after all hooks
  if (!userProfile?.is_onboarded || accounts.length === 0) {
    return (
      <div className="flex flex-col items-center justify-center w-full h-full bg-background gap-8">
        <div className="relative">
          <div className="absolute inset-0 bg-primary/20 blur-[40px] rounded-full animate-pulse"></div>
          <div className="w-20 h-20 rounded-2xl bg-white/5 border border-white/10 flex items-center justify-center relative z-10">
            <LayoutDashboard size={32} className="text-primary animate-pulse" />
          </div>
        </div>
        <div className="flex flex-col items-center gap-2">
          <h2 className="text-xl font-bold tracking-tight text-foreground font-mono text-center">
            Finalizing workspace & syncing transactions...
          </h2>
          <p className="text-sm text-muted-foreground text-center">This will only take a moment.</p>
        </div>
      </div>
    );
  }

  return (
    <div className="flex-1 flex flex-col overflow-hidden w-full">
      {ticker && <div className="w-full shrink-0">{ticker}</div>}
      <div className="flex-1 flex flex-col px-10 pt-10 overflow-hidden w-full max-w-screen-2xl mx-auto">
        <div className="flex items-center justify-between mb-10 shrink-0">
          <div className="flex flex-col gap-4">
            <NLPDashboardController />
          </div>
          <div className="flex items-center gap-6">
            <div className="flex items-center bg-white/5 backdrop-blur-xl border-[0.5px] border-white/10 rounded-full p-1 shadow-inner">
              <Button
                isIconOnly
                onPress={() => setTheme("dark")}
                className={`w-8 h-8 min-w-8 rounded-full border-none flex justify-center items-center transition-all ${
                  mounted && theme === "dark"
                    ? "bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 shadow-lg"
                    : "text-foreground/30 hover:text-foreground bg-transparent"
                }`}
              >
                <Moon size={14} />
              </Button>
              <Button
                isIconOnly
                onPress={() => setTheme("light")}
                className={`w-8 h-8 min-w-8 rounded-full border-none flex justify-center items-center transition-all ${
                  mounted && theme === "light"
                    ? "bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 shadow-lg"
                    : "text-foreground/30 hover:text-foreground bg-transparent"
                }`}
              >
                <Sun size={14} />
              </Button>
            </div>
            <Button
              isIconOnly
              className="w-10 h-10 min-w-10 rounded-full bg-white/5 backdrop-blur-xl border-[0.5px] border-white/10 text-foreground shadow-lg relative cursor-pointer flex justify-center items-center hover:border-primary/50 transition-all"
            >
              <Bell size={16} />
              <span className="absolute top-2.5 right-2.5 w-2 h-2 bg-destructive rounded-full border border-black shadow-[0_0_8px_rgba(239,68,68,0.8)]"></span>
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto scrollbar-hide pb-24 relative">
          <ProactiveInsightsFeed />
          
          {isLoading && localWidgets.length === 0 ? (
            <div className="w-full h-96 flex items-center justify-center">
              <span className="text-primary font-medium tracking-widest uppercase text-xs">
                Synchronizing AI Dashboard State...
              </span>
            </div>
          ) : isError && localWidgets.length === 0 ? (
            <div className="w-full h-96 flex flex-col items-center justify-center text-danger font-medium tracking-widest uppercase text-xs">
              <span>Failed to synchronize dashboard state.</span>
              <span className="text-[10px] mt-2 opacity-50">Please ensure the backend API is running.</span>
            </div>
          ) : (
            <DndContext
              sensors={sensors}
              collisionDetection={closestCenter}
              onDragEnd={handleDragEnd}
            >
              <SortableContext
                items={localWidgets.map((w) => w.widget_uuid)}
                strategy={rectSortingStrategy}
              >
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full">
                  {localWidgets.map((widget) => (
                    <SortableWidgetItem
                      key={widget.widget_uuid}
                      id={widget.widget_uuid}
                      height={widget.localHeight}
                      colSpan={widget.localColSpan}
                      onResize={handleResizeWidget}
                    >
                      {renderWidgetContent(widget)}
                    </SortableWidgetItem>
                  ))}
                </div>
              </SortableContext>
            </DndContext>
          )}

          <div className="w-full flex justify-center mt-12 mb-8">
            <Button
              onPress={() => setIsModalOpen(true)}
              variant="primary"
              className="bg-primary/10 text-primary hover:bg-primary/20 border border-primary/30 shadow-lg rounded-xl flex items-center gap-2 px-6 py-6 transition-all cursor-pointer"
            >
              <Plus size={20} /> Add New Widget
            </Button>
          </div>
        </div>

        {isModalOpen && (
          <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-transparent/80 backdrop-blur-none">
            <div className="bg-black border border-white/10 rounded-3xl w-full max-w-md shadow-2xl overflow-hidden flex flex-col">
              <div className="p-6 border-b border-border flex items-center justify-between">
                <h2 className="text-foreground font-bold text-lg tracking-tight">
                  Add Widget
                </h2>
                <CloseButton
                  onClick={() => setIsModalOpen(false)}
                  className="text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors rounded-lg w-8 h-8 flex items-center justify-center"
                />
              </div>
              <div className="p-4 flex flex-col gap-2 max-h-[60vh] overflow-y-auto scrollbar-hide">
                {AVAILABLE_WIDGET_TYPES.map((widget) => (
                  <Button
                    key={widget.type}
                    variant="primary"
                    onPress={() => handleAddManualWidget(widget.type)}
                    className="w-full h-auto text-left flex justify-start items-center gap-4 p-4 rounded-2xl hover:bg-secondary transition-all border border-transparent hover:border-border group cursor-pointer bg-transparent"
                  >
                    <div className="w-12 h-12 rounded-xl bg-transparent border border-border flex items-center justify-center text-muted-foreground group-hover:text-primary group-hover:bg-primary/10 group-hover:shadow-[0_0_15px_rgba(0,242,255,0.2)] transition-all shrink-0">
                      <widget.icon size={20} />
                    </div>
                    <span className="text-foreground font-medium text-sm tracking-wide">
                      {widget.label}
                    </span>
                  </Button>
                ))}
              </div>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
