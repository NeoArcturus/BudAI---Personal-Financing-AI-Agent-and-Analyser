export function getCurrencySymbol(currency?: string): string {
  if (!currency) return "£";
  const c = currency.toUpperCase().trim();
  switch (c) {
    case "GBP":
      return "£";
    case "EUR":
      return "€";
    case "USD":
      return "$";
    case "JPY":
      return "¥";
    case "CHF":
      return "CHF ";
    case "CAD":
      return "CA$";
    case "AUD":
      return "A$";
    default:
      return `${c} `;
  }
}

export function formatCurrency(amount: number, currency: string = "GBP"): string {
  const symbol = getCurrencySymbol(currency);
  const formatted = Math.abs(amount).toLocaleString("en-GB", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  });
  if (amount < 0) {
    return `-${symbol}${formatted}`;
  }
  return `${symbol}${formatted}`;
}
