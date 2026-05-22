# BudAI Engineering Context & UI Standards

## Project Identity
**BudAI** is an institutional-grade financial intelligence platform. It provides a "Digital Twin" experience for personal finance, focusing on data precision, risk analysis, and asset growth. 

The platform's goal is to present complex financial insights with the clarity and professional tone of a high-tier financial advisor, stripping away all technical jargon and marketing fluff to focus entirely on **Data Integrity**.

---

# Mandatory Terminology Standards

## Prohibited Terms (Banned from UI)
The following terms are strictly forbidden in all user-facing interfaces, components, and documentation. Use the approved alternatives instead.

| Banned Term | Approved Alternative | Rationale |
| :--- | :--- | :--- |
| **Engine** | Tool, Service, Analysis, Advisor | Too technical; implies a machine rather than an advisor. |
| **Model** | Projection, Trend, Calculation | Tech-stack jargon; distracts from financial insights. |
| **Agent / Multi-Agent** | Intelligence, Analysis, Service | Implies external entities; keep focus on unified advisor experience. |
| **Orchestration** | Analysis, Integration | Jargon; confusing for financial users. |
| **Phase X (e.g., Phase 5)** | (Omit or use "Advanced") | Internal roadmap terminology; irrelevant to end-users. |
| **LSTM / Bates / AI** | (Omit or use "Forecast / Insight") | Technical acronyms; maintain "Digital Twin" mystery and focus on data. |
| **Velocity** | Growth, Rate, Yield | Racing metaphor; violates financial sobriety. |
| **Turbo / Nitro / Blazing** | (Omit or use "Instant / Real-time") | Competitive/Racing terminology; unprofessional. |
| **Evolution / Evolutionized** | (Omit or use "Precision / Professional") | Marketing fluff; lacks technical precision. |
| **Node** | Account, Connection, Profile | Graph theory jargon; use standard financial entities. |

## Professional Tone & Voice
- **Institutional-Grade:** Write like a senior wealth manager, not a tech startup.
- **Sobriety:** Avoid exclamation marks, emojis, and superlative marketing claims (e.g., "Blazing fast", "The best ever").
- **Clarity:** Use precise financial terms (Asset, Liability, Yield, Accumulation, Outflow, Inflow).
- **Empathy without Fluff:** The advisor should be supportive but remain mathematically grounded.

---

# Core Architectural Rules

## 1. Data-First Integrity
- **No Hallucinations:** AI outputs must be strictly grounded in verified financial data.
- **"I do not have data on that":** Use this exact phrase (or equivalent professional tone) when data is missing. Do not guess.
- **GBP Only:** All financial values must use the **£** symbol. No exceptions for UI placeholders.

## 2. Liquid State Dashboard Architecture
- **Widget Independence:** Every widget is a self-sustaining entity that fetches its own data and handles its own errors.
- **No Shared Mutable State:** Avoid coupling widgets to a central "Manager" for data; they should rely on isolated API calls.
- **Clean De-cluttering:** If a component or text does not directly serve a financial insight or utility, it should be removed. No placeholders or "dummy stats".

## 3. Security & Connectivity
- **Bank-Level Transparency:** Always emphasize "Read-Only API Access" and "AES-256 Encryption".
- **Status Indicators:** Use grounded status labels like "System Operational" or "Service Unavailable" instead of "Engine Offline".

---

# Engineering Standards

## TypeScript
- **Strict Mode:** Never use `any`.
- **Explicit Interfaces:** Every API response and component prop must have a typed interface.

## CSS & UI (Tailwind v4)
- **CSS-First Config:** Use `app/globals.css` for theming (`@theme`) and utility classes.
- **No Scrollbars:** Always use `scrollbar-hide` utilities to maintain the "App-like" feel.
- **HeroUI Enforcement:** Use HeroUI components for all standard UI elements (Buttons, Cards, Modals). Native elements are only allowed for performance-critical or missing components.

## Performance
- **Component Isolation:** Minimize re-renders by keeping state local to widgets.
- **Incremental Loading:** Use skeletons and loading states within individual widgets rather than blocking the entire page.
