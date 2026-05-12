"use client";

import React, { useState } from "react";
import { useBudAI } from "@/app/context/AppContext";
import CashFlowWidget from "@/app/(protected)/_components/CashFlowWidget";
import SpendingTrendWidget from "@/app/(protected)/_components/SpendingTrendWidget";
import ExpenseDistributionWidget from "@/app/(protected)/_components/ExpenseDistributionWidget";
import PortfolioCardWidget from "@/app/(protected)/_components/PortfolioCardWidget";
import LedgerTableWidget from "@/app/(protected)/_components/LedgerTableWidget";
import {
  Search,
  Bell,
  Settings,
  LayoutDashboard,
  ArrowRightLeft,
  Clock,
  RefreshCcw,
  CreditCard,
  Moon,
  Sun,
  Plus,
  X,
  LineChart,
  BarChart,
  PieChart,
  MessageSquare,
} from "lucide-react";
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
import { Button, Input, Link } from "@heroui/react";
import { cn } from "@/lib/utils";

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

const AVAILABLE_WIDGET_TYPES = [
  { type: "cashFlow", label: "Cash Flow - Income vs Expense", icon: BarChart },
  {
    type: "spendingTrend",
    label: "Spending Trend - Historical",
    icon: LineChart,
  },
  {
    type: "expenseDistribution",
    label: "Expense Distribution",
    icon: PieChart,
  },
  {
    type: "ledger",
    label: "Ledger Table - Historical Transactions",
    icon: Clock,
  },
  { type: "portfolio", label: "Portfolio Card", icon: CreditCard },
  { type: "aiChat", label: "AI Chat Session Headings", icon: MessageSquare },
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

  const handleResizePointerDown = (e: React.PointerEvent) => {
    e.stopPropagation();
    e.preventDefault();

    const startY = e.clientY;
    const startHeight = height;

    const onPointerMove = (moveEvent: PointerEvent) => {
      const deltaY = moveEvent.clientY - startY;
      const newHeight = Math.max(250, startHeight + deltaY);
      onResize(id, newHeight, colSpan);
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
        className="absolute bottom-0 right-0 w-6 h-6 cursor-ns-resize opacity-0 group-hover:opacity-100 transition-opacity z-50 flex items-center justify-center"
      >
        <div className="w-4 h-1 bg-white/30 rounded-full" />
      </div>
    </div>
  );
}

export default function HomePage() {
  const { userName } = useBudAI();
  const [isModalOpen, setIsModalOpen] = useState(false);
  const [theme, setTheme] = useState<"dark" | "light">("dark");

  const [widgets, setWidgets] = useState<WidgetInstance[]>([
    { id: "cashFlow-1", type: "cashFlow", height: 400, colSpan: 1 },
    { id: "portfolio-1", type: "portfolio", height: 280, colSpan: 1 },
    { id: "ledger-1", type: "ledger", height: 500, colSpan: 2 },
    { id: "spendingTrend-1", type: "spendingTrend", height: 400, colSpan: 1 },
    {
      id: "expenseDistribution-1",
      type: "expenseDistribution",
      height: 350,
      colSpan: 1,
    },
  ]);

  const sensors = useSensors(
    useSensor(PointerSensor, {
      activationConstraint: {
        distance: 8,
      },
    }),
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

  const getInitialWidgetParams = (type: string) => {
    if (type === "ledger") return { height: 500, colSpan: 2 };
    if (type === "portfolio") return { height: 280, colSpan: 1 };
    if (type === "expenseDistribution") return { height: 350, colSpan: 1 };
    return { height: 400, colSpan: 1 };
  };

  const handleAddWidget = (type: string) => {
    const params = getInitialWidgetParams(type);
    const newWidget: WidgetInstance = {
      id: `${type}-${crypto.randomUUID()}`,
      type,
      height: params.height,
      colSpan: params.colSpan,
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
    switch (widget.type) {
      case "cashFlow":
        return (
          <CashFlowWidget onRemove={() => handleRemoveWidget(widget.id)} />
        );
      case "spendingTrend":
        return (
          <SpendingTrendWidget onRemove={() => handleRemoveWidget(widget.id)} />
        );
      case "expenseDistribution":
        return (
          <ExpenseDistributionWidget
            onRemove={() => handleRemoveWidget(widget.id)}
          />
        );
      case "portfolio":
        return <PortfolioCardWidget />;
      case "ledger":
        return (
          <LedgerTableWidget onRemove={() => handleRemoveWidget(widget.id)} />
        );
      case "aiChat":
        return (
          <div className="w-full h-full bg-[#13151D]/40 backdrop-blur-xl rounded-3xl border border-[#00E5FF]/20 flex items-center justify-center relative shadow-[0_0_30px_rgba(0,229,255,0.05)]">
            <span className="text-[#00E5FF]/50 font-medium">
              AI Chat Module Offline
            </span>
            <Button
              isIconOnly
              variant="primary"
              onPress={() => handleRemoveWidget(widget.id)}
              className="absolute top-4 right-4 text-[#8B8E98] hover:text-white min-w-8 w-8 h-8 rounded-lg"
            >
              <X size={16} />
            </Button>
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen w-full bg-[#08090D] font-sans overflow-hidden">
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[10%] -left-[5%] w-[70%] h-[70%] rounded-full bg-[#00E5FF]/10 blur-[180px]"></div>
        <div className="absolute -bottom-[10%] -right-[5%] w-[70%] h-[70%] rounded-full bg-[#FF3366]/10 blur-[180px]"></div>
      </div>

      <div className="relative z-10 w-64 h-full bg-[#13151D]/40 backdrop-blur-2xl border-r border-white/5 flex flex-col justify-between py-8 px-6 shrink-0 shadow-[4px_0_30px_rgba(0,0,0,0.3)]">
        <div>
          <div className="flex items-center gap-3 mb-12">
            <div className="w-8 h-8 rounded-lg bg-linear-to-br from-[#00E5FF] to-[#0088FF] flex items-center justify-center shadow-[0_0_15px_rgba(0,229,255,0.4)]">
              <span className="text-[#08090D] font-black text-lg leading-none tracking-tighter">
                B
              </span>
            </div>
            <h1 className="text-white text-2xl font-bold tracking-tight">
              BudAI
            </h1>
          </div>
          <nav className="space-y-2">
            <Link
              href="#"
              className="flex items-center gap-4 text-[#00E5FF] bg-[#00E5FF]/10 px-4 py-3 rounded-xl shadow-[inset_0_0_20px_rgba(0,229,255,0.05)] transition-all border border-[#00E5FF]/20"
            >
              <LayoutDashboard size={20} />
              <span className="font-semibold text-sm">Dashboard</span>
            </Link>
            <Link
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <ArrowRightLeft size={20} />
              <span className="font-medium text-sm">Transactions</span>
            </Link>
            <Link
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <Clock size={20} />
              <span className="font-medium text-sm">History</span>
            </Link>
            <Link
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <RefreshCcw size={20} />
              <span className="font-medium text-sm">Exchange</span>
            </Link>
            <Link
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <CreditCard size={20} />
              <span className="font-medium text-sm">Payments</span>
            </Link>
            <Link
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <Bell size={20} />
              <span className="font-medium text-sm">Notifications</span>
            </Link>
            <Link
              href="#"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all mt-4"
            >
              <Settings size={20} />
              <span className="font-medium text-sm">Settings</span>
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-3 pt-6 border-t border-white/10 mt-8">
          <div className="w-10 h-10 rounded-full bg-linear-to-br from-[#00E5FF] to-[#FF3366] shrink-0 shadow-[0_0_15px_rgba(255,51,102,0.3)] border border-white/10"></div>
          <div className="overflow-hidden">
            <p
              suppressHydrationWarning
              className="text-white text-sm font-semibold truncate"
            >
              {userName || "User"}
            </p>
            <p className="text-[#00E5FF]/70 font-medium text-xs truncate tracking-wide">
              BudAI Member
            </p>
          </div>
        </div>
      </div>

      <div className="relative z-10 flex-1 flex flex-col pt-8 px-8 h-full">
        <div className="flex items-center justify-between mb-8 shrink-0">
          <div className="relative w-75">
            <Search className="absolute left-4 top-1/2 -translate-y-1/2 w-4 h-4 text-[#5E6272] pointer-events-none z-10" />
            <Input
              type="text"
              placeholder="Search ledger..."
              className="bg-[#13151D]/40 backdrop-blur-xl border border-white/5 rounded-xl focus-within:border-[#00E5FF]/50 shadow-[0_4px_20px_rgba(0,0,0,0.3)] [&_input]:pl-11 [&_input]:text-sm [&_input]:text-white [&_input::placeholder]:text-[#5E6272]"
            />
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center bg-[#13151D]/40 backdrop-blur-xl border border-white/5 rounded-full p-1 shadow-lg">
              <Button
                isIconOnly
                variant="primary"
                onPress={() => setTheme("dark")}
                className={cn(
                  "w-9 h-9 min-w-9 rounded-full transition-colors cursor-pointer border-none",
                  theme === "dark"
                    ? "bg-[#00E5FF]/20 text-[#00E5FF]"
                    : "bg-transparent text-[#5E6272] hover:text-white",
                )}
              >
                <Moon size={16} />
              </Button>
              <Button
                isIconOnly
                variant="primary"
                onPress={() => setTheme("light")}
                className={cn(
                  "w-9 h-9 min-w-9 rounded-full transition-colors cursor-pointer border-none",
                  theme === "light"
                    ? "bg-[#FF3366]/20 text-[#FF3366]"
                    : "bg-transparent text-[#5E6272] hover:text-white",
                )}
              >
                <Sun size={16} />
              </Button>
            </div>
            <Button
              isIconOnly
              variant="primary"
              className="w-11 h-11 min-w-11 rounded-full bg-[#13151D]/40 backdrop-blur-xl border border-white/5 text-white shadow-[0_4px_20px_rgba(0,0,0,0.3)] relative cursor-pointer"
            >
              <Bell size={18} />
              <span className="absolute top-3 right-3 w-2.5 h-2.5 bg-[#FF3366] rounded-full border-2 border-[#13151D] shadow-[0_0_8px_rgba(255,51,102,0.8)]"></span>
            </Button>
          </div>
        </div>

        <div className="flex-1 overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none] pb-24 relative">
          <DndContext
            sensors={sensors}
            collisionDetection={closestCenter}
            onDragEnd={handleDragEnd}
          >
            <SortableContext
              items={widgets.map((w) => w.id)}
              strategy={rectSortingStrategy}
            >
              <div className="grid grid-cols-1 lg:grid-cols-2 gap-6 w-full auto-rows-min items-start">
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

                <div
                  onClick={() => setIsModalOpen(true)}
                  className="h-70 w-full rounded-3xl border-2 border-dashed border-white/10 hover:border-[#00E5FF]/50 bg-[#13151D]/20 hover:bg-[#00E5FF]/5 transition-all flex flex-col items-center justify-center gap-4 group cursor-pointer col-span-1 shadow-lg"
                >
                  <div className="w-14 h-14 rounded-2xl bg-[#181A20] border border-white/10 flex items-center justify-center group-hover:scale-110 group-hover:shadow-[0_0_20px_rgba(0,229,255,0.2)] transition-all duration-300">
                    <Plus
                      size={24}
                      className="text-[#5E6272] group-hover:text-[#00E5FF] transition-colors"
                    />
                  </div>
                  <span className="text-[#5E6272] font-semibold tracking-wide uppercase text-xs group-hover:text-[#00E5FF] transition-colors">
                    Add New Module
                  </span>
                </div>
              </div>
            </SortableContext>
          </DndContext>
        </div>
      </div>

      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-xl">
          <div className="bg-[#08090D] border border-white/10 rounded-3xl w-full max-w-md shadow-[0_0_50px_rgba(0,229,255,0.1)] overflow-hidden flex flex-col">
            <div className="p-6 border-b border-white/5 flex items-center justify-between bg-[#13151D]/50">
              <h2 className="text-white font-bold text-lg tracking-tight">
                System Modules
              </h2>
              <Button
                isIconOnly
                variant="primary"
                onPress={() => setIsModalOpen(false)}
                className="text-[#8B8E98] hover:text-[#FF3366] hover:bg-[#FF3366]/10 min-w-8 w-8 h-8 rounded-lg transition-colors border-none bg-transparent cursor-pointer"
              >
                <X size={18} />
              </Button>
            </div>
            <div className="p-4 flex flex-col gap-2 max-h-[60vh] overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
              {AVAILABLE_WIDGET_TYPES.map((widget) => (
                <Button
                  key={widget.type}
                  variant="primary"
                  onPress={() => handleAddWidget(widget.type)}
                  className="w-full h-auto text-left flex justify-start items-center gap-4 p-4 rounded-2xl hover:bg-[#00E5FF]/5 transition-all border border-transparent hover:border-[#00E5FF]/30 group cursor-pointer bg-transparent"
                >
                  <div className="w-12 h-12 rounded-xl bg-[#13151D] border border-white/5 flex items-center justify-center text-[#8B8E98] group-hover:text-[#00E5FF] group-hover:bg-[#00E5FF]/10 group-hover:shadow-[0_0_15px_rgba(0,229,255,0.2)] transition-all shrink-0">
                    <widget.icon size={20} />
                  </div>
                  <span className="text-white font-medium text-sm tracking-wide">
                    {widget.label}
                  </span>
                </Button>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
