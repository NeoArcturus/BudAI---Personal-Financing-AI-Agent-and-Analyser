# BudAI Context & Guidelines

## Project Overview

BudAI is a personal financing AI agent and analyzer. The application utilizes a "Liquid State Dashboard" with "Context-Isolated Modals." It features a high-density, modular frontend using a vertical-scrolling, card-based feed.

## Tech Stack

- **Framework:** Next.js (App Router), React
- **Language:** TypeScript (Strict)
- **Styling:** Tailwind CSS
- **UI Components:** HeroUI v3 (Strictly replace standard HTML tags like `<button>`, `<img>` with `<Button>`, `<Image>`, etc.)
- **Interactions:** `@dnd-kit/core`, `@dnd-kit/sortable` (for widget layout and positioning)
- **Charting:** `chart.js` via a centralized `CoreChartEngine`

## Core Architecture: Liquid State Dashboard

The application follows a multi-page FinTech architecture driven by specialized AI personas:

1. **The Global Liquid Dashboard (The Narrator):** Intelligence feed, forecasting hub, macro pulse. Widgets are isolated and independently fetch data.
2. **Transaction Intelligence Ledger (The Auditor):** Clustered liquid feed, category clusters, anomaly spotlight, isolated audit modals.
3. **Wealth & Health Portfolio (The Wealth Manager):** Survival metrics feed, growth trackers, health radar modal.
4. **Global Strategy & Goals (The Architect):** Goal cards, simulation modal, constraint manager.
5. **Multi-Bank Connection Center (The Integrator):** Connection status feed, re-auth hub.
6. **Command Center (The HUD Overlay):** Semantic search and action routing overlay.

## UI/UX Rules

- **No Scrollbars:** Implement liquid states without visible scrollbars using utility classes like `scrollbar-hide`, `[&::-webkit-scrollbar]:hidden`, `[-ms-overflow-style:none]`, and `[scrollbar-width:none]`.
- **Independent Widgets:** Every widget must be fully independent. They must manage their own state, handle their own data fetching, and include their own HeroUI `<Dropdown>` for account selection.
- **No Forced Layout Restrictions:** Widgets should be resizable and seamlessly integrate into a 2-column or variable grid using `dnd-kit`.

## Strict Coding Guidelines

- **Type Safety:** No `any` type allowed in TypeScript. Define strict interfaces for all data structures and API responses.
- **Data Fetching:** No mock data allowed unless explicitly requested. Always fetch from designated endpoints (e.g., `/api/accounts/{id}/transactions`, `/api/media/execute`).
- **Comments:** Do not write inline comments.
- **Docstrings:** Do not remove existing docstrings for functions. Add docstrings for functions only when they are not present and are absolutely required.
- **Emojis:** Do not use emojis in the code.
- **Terminology Restriction:** Do not use any racing terminology while discussing tech, code, problems, or queries.

## API Integration Patterns

Data fetching should occur at the component level to ensure widget independence.

- **Transactions:** `/api/accounts/{id}/transactions?from={date}&to={date}`
- **AI Tool Execution:** `/api/media/execute` (Requires `tool_name` and `parameters`)
- **AI Explanations:** `/api/chat/explain`
