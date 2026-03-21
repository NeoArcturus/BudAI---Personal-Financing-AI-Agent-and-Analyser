import { Transaction } from "@/types";

interface TransactionFeedProps {
  transactions: Transaction[];
  showCategory: boolean;
}

export default function TransactionFeed({
  transactions,
  showCategory,
}: TransactionFeedProps) {
  if (!transactions || transactions.length === 0) {
    return (
      <div className="text-slate-500 text-sm text-center py-8">
        No transactions found.
      </div>
    );
  }

  return (
    <div className="space-y-3">
      <div className="flex justify-between items-center text-[10px] font-bold tracking-widest text-slate-500 uppercase px-4 pb-2 border-b border-slate-800">
        <div className={showCategory ? "w-1/2" : "w-3/4"}>
          Transaction Details
        </div>
        {showCategory && <div className="w-1/4 text-center">Category</div>}
        <div className="w-1/4 text-right">Amount</div>
      </div>

      {transactions.map((tx, i) => {
        const description = tx.description || tx.Description || "Unknown";
        const dateStr = tx.timestamp || tx.date || "";
        const category = tx.category || tx.Category || "Uncategorized";

        const amount = tx.amount ?? tx.Amount ?? 0;
        const isNegative = amount < 0;
        const displayAmount = Math.abs(amount).toFixed(2);

        return (
          <div
            key={i}
            className="bg-[#1c2128] p-4 rounded-xl border border-slate-800/50 text-sm hover:border-[#00FFAA]/30 transition-colors flex items-center justify-between"
          >
            <div
              className={`flex flex-col ${showCategory ? "w-1/2" : "w-3/4"} pr-4`}
            >
              <span
                className="truncate text-slate-200 font-medium"
                title={description}
              >
                {description}
              </span>
              <span className="text-[10px] text-slate-500 mt-1">{dateStr}</span>
            </div>

            {showCategory && (
              <div className="w-1/4 flex justify-center">
                <span className="bg-[#0D1117] border border-slate-700 px-3 py-1 rounded-full text-[10px] uppercase tracking-wider text-[#00FFAA] font-bold shadow-inner truncate max-w-full text-center">
                  {category === "Uncategorized" ? "" : category}
                </span>
              </div>
            )}

            <div
              className={`w-1/4 text-right font-mono font-bold text-base ${isNegative ? "text-red-400" : "text-[#00FFAA]"}`}
            >
              {isNegative ? "-" : "+"}£{displayAmount}
            </div>
          </div>
        );
      })}
    </div>
  );
}
