"use client";

import React, { useState, useEffect } from "react";
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
import { Button, toast, CloseButton, SearchField } from "@heroui/react";
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
  User,
  Brain,
  Repeat,
  Zap,
  ShieldAlert,
  CalendarDays,
  Activity,
} from "lucide-react";
import { useBudAI } from "@/app/context/AppContext";
import { useTheme } from "next-themes";
import { NLPDashboardController } from "@/components/dashboard/NLPDashboardController";
import { ProactiveInsightsFeed } from "@/components/dashboard/ProactiveInsightsFeed";

interface WidgetInstance {
  id: string;
  type: string;
  height: number;
  colSpan: number;
}

interface SortableWidgetProps {
  id: string;
  height: number;
  colSpan: number;
  onResize: (id: string, newHeight: number, newColSpan: number) => void;
  children: React.ReactNode;
}

export const WidgetContext = React.createContext<{
  onRemove?: () => void;
  instanceId?: string;
}>({});

const AVAILABLE_WIDGET_TYPES = [
  { type: "cashFlow", label: "Cash Flow", icon: BarChart },
  { type: "spendingTrend", label: "Spending Trend", icon: LineChart },
  { type: "expenseDistribution", label: "Expense Distribution", icon: PieChart },
  { type: "commodityMarket", label: "Commodities Market", icon: Globe },
  { type: "financialNews", label: "Financial News", icon: MessageSquare },
  { type: "aiChat", label: "Chat Sessions", icon: MessageSquare },
  { type: "analyticsHealth", label: "Financial Health", icon: Activity },
  { type: "analyticsHabits", label: "Habits", icon: Brain },
  { type: "analyticsSubscriptions", label: "Subscriptions", icon: Repeat },
  { type: "analyticsAnomalies", label: "Anomalies", icon: Zap },
  { type: "analyticsRisk", label: "Liability Health", icon: ShieldAlert },
  { type: "analyticsForecast", label: "30-Day Forecast", icon: CalendarDays },
];

function SortableWidgetItem({
  id,
  height,
  colSpan,
  onResize,
  children,
}: SortableWidgetProps) {
  const { attributes, listeners, setNodeRef, transform, transition } =
    useSortable({ id });

  const style = {
    transform: CSS.Transform.toString(transform),
    transition,
    height: `${height}px`,
    gridColumn: `span ${colSpan} / span ${colSpan}`,
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
      className="relative flex touch-none group"
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

interface DashboardClientProps {
  widgetsMap: Record<string, React.ReactNode>;
  ticker?: React.ReactNode;
}

export default function DashboardClient({
  widgetsMap,
  ticker,
}: DashboardClientProps) {
  const { userName } = useBudAI();
  const { theme, setTheme } = useTheme();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [widgets, setWidgets] = useState<WidgetInstance[]>([]);
  const [loadedUser, setLoadedUser] = useState<string | null>(null);
  const [mounted, setMounted] = useState(false);

  useEffect(() => {
    setMounted(true);
  }, []);

  useEffect(() => {
    if (!userName || userName === "User" || userName === "Verified User") {
      const actualName = localStorage.getItem("budai_user_name");
      if (actualName && actualName !== userName) return;
    }

    const defaultWidgets: WidgetInstance[] = [
      { id: "portfolio-1", type: "portfolio", height: 450, colSpan: 1 },
    ];
    const storageKey = `budai_widgets_dashboard_${userName}`;
    const saved = localStorage.getItem(storageKey);
    if (saved) {
      try {
        setWidgets(JSON.parse(saved));
      } catch (e) {
        console.error("Failed to load widgets:", e);
        toast.danger("Failed to load widgets layout");
        setWidgets(defaultWidgets);
      }
    } else {
      setWidgets(defaultWidgets);
    }
    setLoadedUser(userName);
  }, [userName]);

  useEffect(() => {
    if (loadedUser === userName && userName && userName !== "User" && userName !== "Verified User") {
      const storageKey = `budai_widgets_dashboard_${userName}`;
      localStorage.setItem(storageKey, JSON.stringify(widgets));
    }
  }, [widgets, loadedUser, userName]);

  const sensors = useSensors(
    useSensor(PointerSensor, { activationConstraint: { distance: 8 } }),
    useSensor(KeyboardSensor, {
      coordinateGetter: sortableKeyboardCoordinates,
    }),
  );

  const handleDragEnd = (event: DragEndEvent) => {
    const { active, over } = event;
    if (over && active.id !== over.id) {
      setWidgets((items) => {
        const oldIndex = items.findIndex((w) => w.id === active.id);
        const newIndex = items.findIndex((w) => w.id === over.id);
        return arrayMove(items, oldIndex, newIndex);
      });
    }
  };

  const handleRemoveWidget = (idToRemove: string) => {
    setWidgets((current) => current.filter((w) => w.id !== idToRemove));
  };

  const handleAddWidget = (type: string) => {
    const newWidget: WidgetInstance = {
      id: `${type}-${crypto.randomUUID()}`,
      type,
      height: 450,
      colSpan: 1,
    };
    setWidgets((current) => [...current, newWidget]);
    setIsModalOpen(false);
  };

  const handleResizeWidget = (
    id: string,
    newHeight: number,
    newColSpan: number,
  ) => {
    setWidgets((current) =>
      current.map((w) =>
        w.id === id ? { ...w, height: newHeight, colSpan: newColSpan } : w,
      ),
    );
  };

  const renderWidgetContent = (widget: WidgetInstance) => {
    const widgetElement = widgetsMap[widget.type];
    if (widgetElement) {
      return (
        <WidgetContext.Provider
          value={{ onRemove: () => handleRemoveWidget(widget.id), instanceId: widget.id }}
        >
          {widgetElement}
        </WidgetContext.Provider>
      );
    }

    if (widget.type === "aiChat") {
      return (
        <div className="w-full h-full liquid-glass rounded-3xl flex items-center justify-center relative shadow-xl overflow-hidden">
          <div className="flex flex-col items-center gap-3 text-muted-foreground/40">
            <Sparkles size={32} />
            <span className="text-xs font-bold uppercase tracking-widest">
              Interactive Analysis
            </span>
          </div>
          <CloseButton
            onClick={() => handleRemoveWidget(widget.id)}
            className="absolute top-4 right-4 text-muted-foreground hover:bg-secondary hover:text-foreground transition-colors rounded-2xl"
          />
        </div>
      );
    }
    return null;
  };

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
          {!loadedUser ? (
            <div className="w-full h-96 flex items-center justify-center">
              <span className="text-primary font-medium tracking-widest uppercase text-xs">
                Restoring Workspace...
              </span>
            </div>
          ) : (
            <DndContext
              sensors={sensors}
              collisionDetection={closestCenter}
              onDragEnd={handleDragEnd}
            >
              <SortableContext
                items={widgets.map((w) => w.id)}
                strategy={rectSortingStrategy}
              >
                <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full">
                  {widgets.map((widget) => (
                    <SortableWidgetItem
                      key={widget.id}
                      id={widget.id}
                      height={widget.height}
                      colSpan={widget.colSpan}
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
            <div className="liquid-glass rounded-3xl w-full max-w-md shadow-2xl overflow-hidden flex flex-col font-geist">
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
                    onPress={() => handleAddWidget(widget.type)}
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
