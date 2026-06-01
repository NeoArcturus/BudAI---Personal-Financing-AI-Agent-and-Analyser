# BudAI Client-Side Application - Financial Analysis Interface

BudAI is a high-fidelity frontend interface for personal financial analysis. It utilizes a modular, grid-based dashboard architecture designed for real-time data visualization and decision-support workflows.

---

## Design System and Visual Specification

The interface implements a standardized dark-mode theme designed for clarity and data precision.

- **UI Layering:** High-contrast dark backgrounds (`#0D1516`) with 24px backdrop filters and translucent component layers for visual depth.
- **Semantic Color Palette:** High-visibility Cyan (`#00E5FF`) for positive indicators and growth actions; Magenta (`#FF3366`) for risk assessment and high-priority alerts.
- **Visual Elevation:** Utilizes inset lighting and localized glow effects to establish component hierarchy and visual focus.

---

## Technical Architecture

### 1. Modular Grid Workspace
A dynamic workspace implemented via `@dnd-kit`. Components are draggable, sortable, and resizable, allowing for a configurable layout that adapts to data density requirements.

### 2. Decoupled Widget Architecture
Every dashboard component (e.g., `CashFlowWidget`, `PortfolioCardWidget`) operates as an independent system. Key architectural requirements include:
- Localized state management and internal business logic.
- Isolated data ingestion from specific backend endpoints.
- Independent filter contexts for account and temporal parameters.

### 3. AI Orchestrator Integration
The frontend is optimized for communication with a distributed microservices orchestrator:
- **Asynchronous Streaming:** Real-time data streaming via the Vercel AI SDK.
- **Structured UI Triggers:** The interface processes structured command payloads emitted by the backend to dynamically render data visualizations.
- **Contextual Analysis Pipeline:** Dedicated workflows for generating data-grounded insights from complex financial datasets.

---

## Technology Stack

- **Framework:** Next.js 15+ (App Router)
- **Language:** TypeScript (Strict Mode)
- **Component Library:** [HeroUI v3](https://heroui.com/)
- **State and Layout:** `@dnd-kit/core` and `@dnd-kit/sortable`
- **Styling:** Tailwind CSS 4 (Utilizing CSS design tokens)
- **Data Visualization:** `Chart.js` (Implemented via a modular Chart Engine)
- **Icons:** `Lucide-React`

---

## Application Structure

```text
app/
├── (auth)/                # Authentication workflows (Login/Register)
├── (protected)/           # Authenticated application workspace
│   ├── _components/       # Modular Dashboard components (CashFlow, Ledger, etc.)
│   ├── _utils/            # Centralized Chart builders and utility logic
│   ├── home/              # Primary Dashboard entry point
│   ├── transactions/      # Transaction analysis ledger
│   ├── forecasting/       # Temporal projection and growth analysis
│   ├── health/            # Financial health assessment metrics
│   └── layout.tsx         # Persistent context and layout provider
├── context/               # Global Application State (Session & Orchestration)
└── globals.css            # Global CSS, theme definitions, and utility classes
lib/                       # API clients and utility functions
types/                     # Strict TypeScript definitions for financial models
```

---

## Deployment and Setup

### Prerequisites
- **Node.js:** 20+ (LTS)
- **Backend API:** BudAI Backend Infrastructure running on `http://localhost:8080`

### Installation
```bash
npm install
```

### Local Development
```bash
npm run dev
```

### Production Build
```bash
npm run build
```

---

## Engineering and Quality Standards

- **Component Standardization:** All UI elements must utilize the HeroUI library to ensure design consistency and accessibility. Native HTML elements are restricted to performance-critical edge cases.
- **Visual Presentation:** Scrollbar visibility is suppressed globally via utility classes to maintain interface consistency.
- **Type Integrity:** Zero-tolerance policy for `any`. All API schemas and component properties must be explicitly defined in `/types/index.ts`.
- **Inversion of Control:** Components must maintain independent data-fetching lifecycles and avoid coupling to shared mutable state or centralized fetch managers.
