# BudAI Frontend

The frontend for the BudAI personal finance application. It provides an interface for tracking finances and interacting with an AI advisor.

## Features
- **Dashboard:** A drag-and-drop grid layout where users can organize financial widgets.
- **Independent Widgets:** Each widget handles its own data fetching directly from the backend.
- **AI Chat:** Integrates with the Vercel AI SDK to stream chat responses, including separate reasoning steps.
- **AI-Driven Onboarding:** Uses LLM analysis during onboarding to automatically set up the best dashboard layout for the user's financial profile.

## Tech Stack
- **Framework:** Next.js 15+ (App Router)
- **Language:** TypeScript
- **UI Components:** HeroUI v3
- **Layout Management:** `@dnd-kit/core` and `@dnd-kit/sortable`
- **Styling:** Tailwind CSS 4
- **Data Visualization:** Chart.js 

## Directory Structure

```text
budai-frontend/
├── app/
│   ├── (auth)/                # Login page
│   ├── (protected)/           # Authenticated application workspace
│   │   ├── _components/       # Shared dashboard UI components (Navbars, Cards, Widgets)
│   │   ├── advisor/           # AI chat and session history
│   │   ├── connections/       # Bank connection management
│   │   ├── home/              # Primary dashboard workspace
│   │   └── layout.tsx         # Persistent context and layout provider
│   ├── context/               # Global state (Session, Auth, User Context)
│   └── onboarding/            # User onboarding flow
├── lib/                       # API client wrappers and hooks
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
- **Strict TypeScript:** Do not use `any` types. Ensure all API responses are typed.
- **Component Library:** Use HeroUI for standard UI elements instead of native HTML elements.
- **State Management:** Components should handle their own API requests independently.
- **Vercel AI SDK:** Streaming chat logic maps custom backend properties directly to Vercel's schema.
