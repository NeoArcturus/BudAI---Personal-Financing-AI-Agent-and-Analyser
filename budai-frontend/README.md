# BudAI Frontend

The frontend client for the BudAI personal finance application. It provides an interface for financial tracking, transaction categorization, and an AI advisory chat to analyze user data.

## Core Capabilities
- **Modular Dashboard:** A grid-based layout using `@dnd-kit` where users can drag, drop, resize, and organize financial widgets.
- **Isolated Component State:** Each dashboard widget independently fetches its own data from the FastAPI backend, decoupling the rendering lifecycle.
- **AI Chat & Reasoning:** Integrates with the Vercel AI SDK to stream real-time tokens from a custom LangGraph backend. It natively supports rendering separated `<think>`/reasoning tokens distinct from the main response.
- **Transaction Ledger:** Includes features for filtering, searching, and overriding machine-learning categorizations for transactions.

## Tech Stack
- **Framework:** Next.js 15+ (App Router)
- **Language:** TypeScript
- **UI Components:** HeroUI v3
- **Layout Management:** `@dnd-kit/core` and `@dnd-kit/sortable`
- **Styling:** Tailwind CSS 4 (with standard CSS utility overrides)
- **Data Visualization:** Chart.js 

## Directory Structure

```text
budai-frontend/
├── app/
│   ├── (auth)/                # Login and registration routing
│   ├── (protected)/           # Authenticated application workspace
│   │   ├── _components/       # Shared dashboard UI components (Navbars, Cards)
│   │   ├── _utils/            # Chart builders and shared formatting logic
│   │   ├── advisor/           # AI chat, session history, and reasoning UI
│   │   ├── connections/       # Bank connection management
│   │   ├── forecasting/       # Financial projections and Chart.js graphs
│   │   ├── health/            # Financial health metrics
│   │   ├── home/              # Primary dashboard workspace
│   │   ├── transactions/      # Transaction ledger and data grid
│   │   └── layout.tsx         # Persistent context and layout provider
│   ├── context/               # Global state (Session, JWT, User Context)
│   └── globals.css            # Global CSS, overrides, and Tailwind configuration
├── lib/                       # API client wrappers and utility functions
├── types/                     # Shared TypeScript definitions
└── tailwind.config.ts         # Tailwind CSS configuration
```

## Local Setup

**Requirements:** Node.js 20+ and the BudAI FastAPI backend running locally on port 8080.

### Environment Setup
Create a `.env.local` file in the root of `budai-frontend`:
```env
NEXT_PUBLIC_API_URL=http://localhost:8080
```

### Installation & Development
```bash
npm install
npm run dev
```

### Production Build
```bash
npm run build
npm start
```

## Engineering Guidelines
- **Strict TypeScript:** Do not use `any` types. Ensure all API responses are explicitly typed in `/types` or directly within the file.
- **Component Library:** Use HeroUI for all standard UI elements. Avoid native HTML elements unless strictly necessary for performance or layout edge-cases.
- **State Management:** Avoid global state for fetching. Components should handle their own API requests and loading states independently to prevent cascading re-renders.
- **Vercel AI SDK Integration:** Streaming chat logic should map custom backend properties (like `reasoning_content`) strictly to Vercel's `parts` or `annotations` schema.
