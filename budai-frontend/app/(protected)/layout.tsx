import { BudAIProvider } from "@/app/context/AppContext";

export default function ProtectedLayout({
  children,
}: {
  children: React.ReactNode;
}) {
  return <BudAIProvider>{children}</BudAIProvider>;
}
