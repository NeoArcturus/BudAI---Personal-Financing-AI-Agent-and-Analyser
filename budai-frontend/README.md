# BudAI Frontend

BudAI Frontend is the interactive user interface for the BudAI personal finance assistant. It provides users with a comprehensive dashboard to securely link bank accounts, visualize spending habits through dynamic analytics charts, and engage with an integrated AI chat panel for real-time financial insights and forecasting.

This is the frontend app for BudAI, built with `Next.js 16, React 19, Tailwind CSS 4` and `Chart.js`.

## Tech Stack

- Next.js 16 (App Router)
- React 19
- TypeScript 5
- Tailwind CSS 4
- Chart.js

## Prerequisites

- Node.js 20+ recommended
- npm
- BudAI backend running on `http://localhost:8080`

## Getting Started

From this folder (`budai-frontend`):

```bash
npm install
npm run dev
```

Open `http://localhost:3000` in your browser.

## Available Scripts

- `npm run dev` - start local dev server
- `npm run build` - build for production
- `npm run start` - start production server
- `npm run lint` - run ESLint

## Backend Dependency

The frontend currently relies on the FastAPI backend running locally on port `8080`. API calls are configured to hit the following base endpoints:

- **Auth & TrueLayer:** `http://localhost:8080/api/auth/...` (Login, Token Exchange, Connection Revocation)
- **Banking Data:** `http://localhost:8080/api/accounts/...` (Fetching balances and transaction history)
- **Conversational AI:** `http://localhost:8080/api/chat/` (Streaming the multi-agent graph responses)
- **Analytics & Visualizations:** `http://localhost:8080/api/media/execute` (Triggering cache generation for charts and radar data)

**Crucial:** You must start the backend server (`uvicorn main:app --port 8080`) _before_ interacting with the frontend. Login, TrueLayer bank linking, chat functionality, and dynamic chart rendering will fail if the backend is unreachable.

> **Note for Production:** While these URLs are currently hardcoded for local development, it is recommended to eventually extract the base URL into a `.env.local` variable (e.g., `NEXT_PUBLIC_API_URL`) to allow seamless transitioning between local testing and live production environments.

## Key Frontend Routes

The application uses the Next.js App Router. Routes are broadly split into public authentication flows and private, protected dashboard views.

### Public Routes

- `/` - **Landing Page:** The initial entry point of the application, introducing BudAI and directing users to authenticate.
- `/login` (via `app/(auth)/login`) - **Authentication:** Existing user login. Upon success, the `budai_token` is saved to browser local storage.
- `/register` (via `app/(auth)/register`) - **Onboarding:** New user registration and initial account setup.

### Protected Routes (`app/(protected)/`)

All routes within the `(protected)` group share a common layout (`layout.tsx`). This layout ensures that the Left Navigation Sidebar and the right-side BudAI Chat panel are persistently visible across all dashboard views.

- `/home` - **Main Dashboard:** The primary landing view post-login. Displays high-level statistic cards (`StatCard.tsx`), a feed of recent bank transactions (`TransactionFeed.tsx`), and quick actions (`QuickActions.tsx`).
- `/finances` - **Analytics & Forecasting:** A deep dive into the user's financial data. This route houses the dynamic visual charts (`ChartDisplay.tsx`, `DynamicChart.tsx`) for categorized spending, cash flow, and future balance predictions.
- `/profile` - **User Management:** Displays user account details, preferences, and the status of linked bank accounts (TrueLayer connections).

## Project Structure

```text
budai-frontend/
├── app/
│   ├── (auth)/
│   │   ├── login/
│   │   └── register/
│   ├── (protected)/
│   │   ├── _components/
│   │   │   ├── BudAIChat.tsx
│   │   │   ├── ChartDisplay.tsx
│   │   │   ├── DynamicChart.tsx
│   │   │   ├── QuickActions.tsx
│   │   │   ├── SidebarLeft.tsx
│   │   │   ├── SidebarRight.tsx
│   │   │   ├── StatCard.tsx
│   │   │   ├── TransactionFeed.tsx
│   │   │   ├── TransactionsControl.tsx
│   │   │   └── TransactionsModal.tsx
│   │   ├── _utils/
│   │   │   └── ChartBuilder.tsx
│   │   ├── finances/
│   │   │   └── page.tsx
│   │   ├── home/
│   │   │   └── page.tsx
│   │   ├── profile/
│   │   │   └── page.tsx
│   │   └── layout.tsx
│   ├── context/
│   │   └── AppContext.tsx
│   ├── favicon.ico
│   ├── globals.css
│   ├── layout.tsx
│   └── page.tsx
├── components/
│   └── ui/
│       ├── Badge.tsx
│       ├── button.tsx
│       ├── card.tsx
│       ├── dialog.tsx
│       ├── input.tsx
│       ├── scroll-area.tsx
│       └── skeleton.tsx
├── lib/
│   │── api.ts
│   └── utils.ts
├── node_modules/
├── public/
├── types/
├── .env
├── .gitignore
├── components.json
├── eslint.config.mjs
├── next-env.d.ts
├── next.config.ts
├── package-lock.json
├── package.json
├── postcss.config.mjs
├── README.md
├── tailwind.config.ts
└── tsconfig.json
```

---

## Notes

- **Authentication:** The session token is stored securely in the browser's local storage as `budai_token`. It acts as a Bearer token and must be attached to the `Authorization` header for all protected backend requests.
- **Global State:** The frontend utilizes React Context (`context/AppContext.tsx`) to manage global app state, including user session data and active TrueLayer bank connections.
- **Persistent Layout:** The `app/(protected)/layout.tsx` enforces a persistent dashboard shell. The Left Navigation handles page routing, while the right-side BudAI Chat panel remains continuously mounted. This ensures that the user's conversational context and chat history are never lost when navigating between the home and finances views.
- **Dynamic UI Rendering:** The chat component is programmed to intercept specific trigger tags sent by the backend graph orchestrator (e.g., `[TRIGGER_CASH_FLOW_CHART:CACHE_123]`). When detected, the frontend strips the raw text tag and dynamically renders the corresponding `Chart.js` component using the provided cache ID.
- **Environment Configuration:** If the backend API host changes from `localhost:8080` (e.g., during deployment), ensure you update the fetch URLs across the frontend services. Migrating these hardcoded URLs to a `.env.local` file (e.g., using `NEXT_PUBLIC_API_URL`) is recommended for production.
