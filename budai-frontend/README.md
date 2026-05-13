# BudAI Frontend - Financial Intelligence Interface

BudAI is a high-fidelity, AI-powered personal finance intelligence platform. It features a **Liquid State Dashboard Architecture** designed for modularity, real-time data visualization, and autonomous financial decision-making support.

---

## Visual Identity: "Obsidian Glass"

The BudAI interface follows a **Cyber-Financial** aesthetic known as **Obsidian Glass**. It prioritizes depth, precision, and futuristic control through:

- **Surface Strategy:** Deep obsidian backgrounds (`#0D1516`) paired with 24px backdrop blurs and semi-transparent glass layers.
- **Chromatic Functionalism:** Neon Cyan (`#00E5FF`) for growth and action; Deep Pink (`#FF3366`) for risk and attention.
- **Elevation through Light:** Inset glows and soft neon halos replace traditional drop shadows to simulate a premium hardware interface.

---

## Core Architecture

### 1. Liquid State Dashboard
A fully dynamic workspace powered by `@dnd-kit`. Widgets are not fixed; they are draggable, sortable, and resizable. The dashboard layout is "liquid," adapting in real-time to user priority and data density.

### 2. Context-Isolated Widgets
In strict adherence to architectural mandates, every widget (e.g., `CashFlowWidget`, `PortfolioCardWidget`) is an independent system. They manage:
- Their own state and business logic.
- Isolated data fetching from dedicated endpoints.
- Independent account-selection and date-filtering contexts.

### 3. AI-Native Integration
The frontend is built to communicate with a multi-agent backend orchestrator:
- **Streaming Intelligence:** Real-time chat with "Neural Core" persona.
- **Semantic Triggers:** The AI can "emit" visualization commands (e.g., `[TRIGGER_CHART]`), which the interface intercepts to render interactive data on the fly.
- **Explanation Engine:** Dedicated pipeline for generating AI-driven insights on complex financial datasets.

---

## Tech Stack

- **Framework:** Next.js 15+ (App Router)
- **Language:** TypeScript (Strict Mode)
- **UI Library:** [HeroUI v3](https://heroui.com/) (formerly NextUI)
- **Layout & Drag:** `@dnd-kit/core` & `@dnd-kit/sortable`
- **Styling:** Tailwind CSS 4 (with CSS Variables for Obsidian Glass tokens)
- **Charting:** `Chart.js` (Centralized via `CoreChartEngine`)
- **Icons:** `Lucide-React`

---

## Project Structure

```text
app/
├── (auth)/                # Context-isolated login/register flows
├── (protected)/           # Authenticated dashboard workspace
│   ├── _components/       # Modular Dashboard Widgets (CashFlow, Ledger, etc.)
│   ├── _utils/            # Centralized Chart Builders and visual logic
│   ├── home/              # Liquid Dashboard entry point
│   ├── transactions/      # Transaction Intelligence Ledger
│   ├── forecasting/       # Wealth & Growth tracking
│   ├── health/            # Financial Health Radar
│   └── layout.tsx         # Persistent context provider
├── context/               # Global AppContext (Session & AI Orchestration)
└── globals.css            # Obsidian Glass global definitions & scrollbar-hide
lib/                       # API clients and utility functions
types/                     # Strict TypeScript interfaces for financial models
```

---

## Getting Started

### Prerequisites
- **Node.js:** 20+ (LTS)
- **Backend:** [BudAI Backend](https://github.com/...) running on `http://localhost:8080`

### Installation
```bash
npm install
```

### Development
```bash
npm run dev
```

### Build
```bash
npm run build
```

---

## Engineering Standards

- **HeroUI Enforcement:** Native HTML elements are avoided; all UI components leverage HeroUI for design consistency.
- **Scrollbar Policy:** All visible scrollbars are hidden using utility classes (`scrollbar-hide`) to maintain the "app-like" aesthetic.
- **Type Safety:** Zero usage of `any`. All API responses and component props must be explicitly typed in `/types/index.ts`.
- **Widget Independence:** Widgets must never share mutable state or rely on parent-level fetch managers.
