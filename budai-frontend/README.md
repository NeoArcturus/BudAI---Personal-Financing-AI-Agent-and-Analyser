# BudAI Frontend

This is the frontend app for BudAI, built with Next.js 16, React 19, Tailwind CSS 4, and Chart.js.

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

The frontend currently calls the backend with hardcoded URLs pointing to:

- `http://localhost:8080/api/auth/...`
- `http://localhost:8080/api/accounts/...`
- `http://localhost:8080/api/chat/`
- `http://localhost:8080/api/media/execute`

Make sure the backend is up before using login, bank linking, account views, chat, or analytics charts.

## Key Frontend Routes

- `/` - landing/auth entry
- `/login` - user login
- `/register` - user registration
- Protected pages under `app/(protected)/` (dashboard and finance views)

## Project Structure

```text
budai-frontend/
├── app/
│   ├── (auth)/
│   ├── (protected)/
│   ├── context/
│   ├── layout.tsx
│   └── page.tsx
├── public/
├── package.json
├── next.config.ts
└── tsconfig.json
```

## Notes

- Auth token is stored in browser local storage as `budai_token`.
- Protected layout includes left navigation + right-side BudAI chat panel.
- If backend API host changes, update frontend fetch URLs accordingly.
