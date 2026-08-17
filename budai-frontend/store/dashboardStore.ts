import { create } from 'zustand';

export interface WidgetInstance {
  id: string;
  type: string;
  height: number;
  colSpan: number;
}

interface DashboardFilters {
  time_type: string;
  from_date?: string;
  to_date?: string;
  category?: string;
  account_ids: string[];
}

interface DashboardState {
  filters: DashboardFilters;
  setFilters: (filters: Partial<DashboardFilters>) => void;
  layout: WidgetInstance[] | null;
  setLayout: (layout: WidgetInstance[]) => void;
}

export const useDashboardStore = create<DashboardState>((set) => ({
  filters: {
    time_type: 'monthly',
    account_ids: ['ALL'],
  },
  setFilters: (newFilters) =>
    set((state) => ({ filters: { ...state.filters, ...newFilters } })),
  layout: null,
  setLayout: (layout) => set({ layout }),
}));
