# BudAI Development Context & Engineering Instructions

## Project Identity

**BudAI** is an AI-powered personal finance intelligence platform focused on:

- Financial analysis
- Transaction intelligence
- Wealth tracking
- Forecasting
- AI-assisted financial decision making

The platform follows a **Liquid State Dashboard Architecture** with **Context-Isolated Modals**, emphasizing modularity, responsiveness, and independent widget systems.

---

# Core Product Architecture

## Primary System Design

BudAI is structured around multiple AI-driven financial workspaces:

### 1. Global Liquid Dashboard — _The Narrator_

#### Purpose

- Financial intelligence feed
- Forecasting center
- Macro-level financial insights
- AI-generated summaries

#### Rules

- Widgets operate independently
- Widgets fetch their own data
- Widgets maintain isolated state management

---

### 2. Transaction Intelligence Ledger — _The Auditor_

#### Purpose

- Transaction clustering
- Category intelligence
- Spending anomaly detection
- Transaction audit workflows

#### Rules

- Cluster-based rendering
- Context-isolated audit modals
- Dynamic filtering and segmentation

---

### 3. Wealth & Health Portfolio — _The Wealth Manager_

#### Purpose

- Financial health scoring
- Asset growth tracking
- Portfolio health radar
- Risk monitoring

#### Rules

- Real-time visualization support
- Independent health metric widgets
- Predictive insights integration

---

### 4. Global Strategy & Goals — _The Architect_

#### Purpose

- Financial goal planning
- Scenario simulation
- Budget constraint modeling
- Long-term strategy generation

#### Rules

- Goal cards must be modular
- Simulation systems must be isolated
- Constraint calculations should remain deterministic

---

### 5. Multi-Bank Connection Center — _The Integrator_

#### Purpose

- Bank account integrations
- Connection management
- Re-authentication workflows
- Account synchronization status

#### Rules

- Connection states must be resilient
- Re-authentication flows must remain isolated
- Bank integrations should support scalability

---

### 6. Command Center — _The HUD Overlay_

#### Purpose

- Semantic search
- AI command routing
- Universal action interface
- Cross-dashboard navigation

#### Rules

- Must remain globally accessible
- Should support contextual execution
- Overlay interactions must not block dashboard rendering

---

# Technology Stack

## Frontend

- **Framework:** Next.js (App Router)
- **Language:** TypeScript (Strict Mode)
- **UI Library:** React
- **Styling:** Tailwind CSS
- **Component Library:** HeroUI v3

## Drag & Layout System

- `@dnd-kit/core`
- `@dnd-kit/sortable`

## Charting

- `chart.js`
- Centralized rendering through `CoreChartEngine`

---

# Mandatory UI & UX Rules

## HeroUI Enforcement

Strictly use HeroUI components whenever applicable.

### Required Replacements

Avoid native HTML elements when HeroUI alternatives exist.

#### Examples

- `<button>` → `<Button>`
- `<img>` → `<Image>`
- `<input>` → HeroUI Input components
- Native modal implementations → HeroUI Modal components

#### Native Elements Are Allowed Only When

- No HeroUI equivalent exists
- Accessibility requires it
- Performance optimization requires it

---

## Scrollbar Policy

Visible scrollbars are prohibited.

### Required Utility Classes

```tsx
scrollbar-hide
[&::-webkit-scrollbar]:hidden
[-ms-overflow-style:none]
[scrollbar-width:none]
```

### Requirements

- Preserve scrolling functionality
- Maintain touch and wheel scrolling
- Ensure accessibility is not degraded

---

## Widget Independence

Every widget must:

- Manage its own state
- Fetch its own data
- Handle its own loading and error states
- Maintain isolated business logic
- Include its own account-selection dropdown when required

Widgets must NOT:

- Depend on parent widget state
- Share mutable state unnecessarily
- Rely on centralized rendering logic for business operations

---

## Layout System Rules

The dashboard must support:

- Resizable widgets
- Variable grid layouts
- Dynamic rearrangement
- Responsive scaling

### Requirements

- Use `dnd-kit`
- Avoid hardcoded sizing constraints
- Maintain smooth drag interactions
- Preserve mobile responsiveness

---

# TypeScript & Engineering Standards

## Strict Type Safety

### Rules

- Never use `any`
- Avoid unsafe casting
- Define explicit interfaces for:
  - API responses
  - Component props
  - State objects
  - Tool parameters
  - Financial models

### Preferred Pattern

```ts
interface Transaction {
  id: string;
  amount: number;
  category: string;
}
```

### Avoid

```ts
const data: any = response;
```

---

## Data Fetching Rules

### No Mock Data

Mock data is prohibited unless explicitly requested.

Always fetch from real endpoints.

---

# Approved API Patterns

## Transactions

```txt
/api/accounts/{id}/transactions?from={date}&to={date}
```

## AI Tool Execution

```txt
/api/media/execute
```

### Required Payload

```json
{
  "tool_name": "string",
  "parameters": {}
}
```

## AI Explanations

```txt
/api/chat/explain
```

---

# Component-Level Fetching

All widgets must fetch their own data.

### Avoid

- Large centralized fetch managers
- Global dependency chains
- Tight coupling between widgets

### Preferred

```tsx
useEffect(() => {
  fetchData();
}, []);
```

inside the widget/component itself.

---

# Code Quality Rules

## Comments

Do not add inline comments unless explicitly requested.

### Avoid

```ts
// fetch user data
```

---

## Docstrings

### Rules

- Never remove existing docstrings
- Add docstrings only when necessary
- Keep docstrings concise and technical

---

## Naming & Terminology Restrictions

Do NOT use:

- Racing terminology
- Racing metaphors
- Competitive speed analogies

### Examples to Avoid

- “turbo mode”
- “nitro optimization”

Use precise engineering terminology instead.

---

## Emoji Policy

Do not use emojis:

- In code
- In documentation
- In variable names
- In generated outputs

---

# State Management Principles

## Preferred Characteristics

State systems should be:

- Predictable
- Localized
- Isolated
- Serializable when possible

### Avoid

- Deep prop drilling
- Excessive global state
- Shared mutable structures

---

# Charting Standards

All chart rendering must:

- Flow through `CoreChartEngine`
- Use strongly typed datasets
- Support responsiveness
- Handle empty, loading, and error states

### Avoid

- Direct unmanaged Chart.js instances
- Duplicate chart configuration logic

---

# Performance Expectations

The application should prioritize:

- Incremental rendering
- Component isolation
- Low re-render frequency
- Efficient data fetching
- Responsive drag interactions

### Avoid

- Monolithic rendering trees
- Heavy synchronous operations in render cycles
- Unnecessary context-wide updates

---

# Expected Development Behavior

When generating or modifying code:

## Always

- Preserve architectural consistency
- Maintain strict typing
- Follow HeroUI conventions
- Keep widgets isolated
- Respect existing design systems
- Use production-grade patterns

## Never

- Introduce mock implementations without request
- Use `any`
- Break widget independence
- Add unnecessary abstractions
- Introduce unrelated libraries
- Add unnecessary inline comments

---

# Output Expectations for Generated Code

Generated code should:

- Be production-ready
- Be fully typed
- Be modular
- Follow BudAI architectural principles
- Integrate cleanly with existing systems
- Avoid placeholder implementations unless explicitly requested
