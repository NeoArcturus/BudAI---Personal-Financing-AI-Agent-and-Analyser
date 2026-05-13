// app/(protected)/home/page.tsx
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
import { Button, Input, Link, Avatar, CloseButton } from "@heroui/react";

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
      const newHeight = Math.max(300, startHeight + deltaY);

      const newColSpan = newHeight > 500 ? 2 : 1;
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
        <div className="w-3 h-3 border-r-2 border-b-2 border-white/30 rounded-br-sm" />
      </div>
    </div>
  );
}

export default function HomePage() {
  const { userName } = useBudAI();
  const [isModalOpen, setIsModalOpen] = useState(false);

  const [widgets, setWidgets] = useState<WidgetInstance[]>([
    { id: "portfolio-1", type: "portfolio", height: 450, colSpan: 1 },
    { id: "cashFlow-1", type: "cashFlow", height: 450, colSpan: 1 },
    { id: "ledger-1", type: "ledger", height: 450, colSpan: 2 },
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
          <div className="w-full h-full bg-obsidian/40 backdrop-blur-[24px] rounded-3xl border border-neon-cyan/20 flex items-center justify-center relative shadow-[0_0_30px_rgba(0,229,255,0.05)]">
            <span className="text-neon-cyan/50 font-medium">
              {widget.type} placeholder (To be implemented)
            </span>
            <CloseButton
              onClick={() => handleRemoveWidget(widget.id)}
              className="text-[#8B8E98] hover:bg-white/10 hover:text-white transition-colors rounded-2xl"
            />
          </div>
        );
      default:
        return null;
    }
  };

  return (
    <div className="flex h-screen w-full bg-obsidian font-geist overflow-hidden">
      <div className="absolute inset-0 z-0 pointer-events-none overflow-hidden">
        <div className="absolute -top-[10%] -left-[5%] w-[70%] h-[70%] rounded-full bg-neon-cyan/10 blur-[180px]"></div>
        <div className="absolute -bottom-[10%] -right-[5%] w-[70%] h-[70%] rounded-full bg-deep-pink/10 blur-[180px]"></div>
      </div>

      <div className="relative z-10 w-64 h-full bg-obsidian/40 backdrop-blur-[24px] border-r border-white/8 flex flex-col justify-between py-8 px-6 shrink-0 shadow-[4px_0_24px_rgba(0,0,0,0.2)]">
        <div>
          <div className="flex items-center gap-3 mb-12">
            <div className="w-8 h-8 rounded-lg bg-linear-to-br from-neon-cyan to-[#0088FF] flex items-center justify-center shadow-[0_0_15px_rgba(0,229,255,0.4)]">
              <span className="text-obsidian font-black text-lg leading-none tracking-tighter">
                B
              </span>
            </div>
            <h1 className="text-white text-2xl font-bold tracking-tight">
              BudAI
            </h1>
          </div>
          <nav className="space-y-2">
            <Link
              href="/home"
              className="flex items-center gap-4 text-neon-cyan bg-neon-cyan/10 px-4 py-3 rounded-xl shadow-[inset_0_0_20px_rgba(0,229,255,0.05)] transition-all border border-neon-cyan/20"
            >
              <LayoutDashboard size={20} />
              <span className="font-semibold text-sm">Dashboard</span>
            </Link>
            <Link
              href="/transactions"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <ArrowRightLeft size={20} />
              <span className="font-medium text-sm">Transactions</span>
            </Link>
            <Link
              href="/forecasting"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <LineChart size={20} />
              <span className="font-medium text-sm">Forecasting</span>
            </Link>
            <Link
              href="/health"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <RefreshCcw size={20} />
              <span className="font-medium text-sm">Health Radar</span>
            </Link>
            <Link
              href="/connections"
              className="flex items-center gap-4 text-[#8B8E98] hover:text-white hover:bg-white/5 px-4 py-3 rounded-xl transition-all"
            >
              <CreditCard size={20} />
              <span className="font-medium text-sm">Connections</span>
            </Link>
          </nav>
        </div>

        <div className="flex items-center gap-3 pt-6 border-t border-white/10 mt-8">
          <Avatar className="w-10 h-10 bg-linear-to-br from-neon-cyan to-deep-pink shrink-0 shadow-[0_0_15px_rgba(255,51,102,0.3)] border border-white/10 text-white font-bold" />
          <div className="overflow-hidden">
            <p
              suppressHydrationWarning
              className="text-white text-sm font-semibold truncate"
            >
              {userName || "User"}
            </p>
            <p className="text-neon-cyan/70 font-medium text-xs truncate tracking-wide">
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
              className="bg-obsidian/40 backdrop-blur-[24px] border border-white/8 rounded-xl focus-within:border-neon-cyan/50 shadow-[0_4px_20px_rgba(0,0,0,0.3)] [&_input]:pl-11 [&_input]:text-sm [&_input]:text-white [&_input::placeholder]:text-[#5E6272]"
            />
          </div>
          <div className="flex items-center gap-4">
            <div className="flex items-center bg-obsidian/40 backdrop-blur-[24px] border border-white/8 rounded-full p-1 shadow-lg">
              <Button
                isIconOnly
                variant="primary"
                className="w-9 h-9 min-w-9 rounded-full bg-neon-cyan/20 text-neon-cyan border-none"
              >
                <Moon size={16} />
              </Button>
              <Button
                isIconOnly
                variant="primary"
                className="w-9 h-9 min-w-9 rounded-full text-[#5E6272] hover:text-white bg-transparent border-none"
              >
                <Sun size={16} />
              </Button>
            </div>
            <Button
              isIconOnly
              variant="primary"
              className="w-11 h-11 min-w-11 rounded-full bg-obsidian/40 backdrop-blur-[24px] border border-white/8 text-white shadow-[0_4px_20px_rgba(0,0,0,0.3)] relative cursor-pointer"
            >
              <Bell size={18} />
              <span className="absolute top-3 right-3 w-2.5 h-2.5 bg-deep-pink rounded-full border-2 border-obsidian shadow-[0_0_8px_rgba(255,51,102,0.8)]"></span>
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

          <div className="w-full flex justify-center mt-12 mb-8">
            <Button
              onPress={() => setIsModalOpen(true)}
              variant="primary"
              className="bg-neon-cyan/10 text-neon-cyan hover:bg-neon-cyan/20 border border-neon-cyan/30 shadow-lg rounded-xl flex items-center gap-2 px-6 py-6 transition-all"
            >
              <Plus size={20} /> Add New Widget
            </Button>
          </div>
        </div>
      </div>

      {isModalOpen && (
        <div className="fixed inset-0 z-50 flex items-center justify-center p-4 bg-black/80 backdrop-blur-[24px]">
          <div className="bg-obsidian border border-white/8 rounded-3xl w-full max-w-md shadow-[0_0_50px_rgba(0,229,255,0.1)] overflow-hidden flex flex-col font-geist">
            <div className="p-6 border-b border-white/8 flex items-center justify-between bg-obsidian/50">
              <h2 className="text-white font-bold text-lg tracking-tight">
                Add Widget
              </h2>

              <CloseButton
                onClick={() => setIsModalOpen(false)}
                className="text-[#8B8E98] hover:text-deep-pink hover:bg-deep-pink/10 transition-colors rounded-lg w-8 h-8 flex items-center justify-center"
              />
            </div>
            <div className="p-4 flex flex-col gap-2 max-h-[60vh] overflow-y-auto [&::-webkit-scrollbar]:hidden [-ms-overflow-style:none] [scrollbar-width:none]">
              {AVAILABLE_WIDGET_TYPES.map((widget) => (
                <Button
                  key={widget.type}
                  variant="primary"
                  onPress={() => handleAddWidget(widget.type)}
                  className="w-full h-auto text-left flex justify-start items-center gap-4 p-4 rounded-2xl hover:bg-neon-cyan/5 transition-all border border-transparent hover:border-neon-cyan/30 group cursor-pointer bg-transparent"
                >
                  <div className="w-12 h-12 rounded-xl bg-obsidian border border-white/8 flex items-center justify-center text-[#8B8E98] group-hover:text-neon-cyan group-hover:bg-neon-cyan/10 group-hover:shadow-[0_0_15px_rgba(0,229,255,0.2)] transition-all shrink-0">
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
