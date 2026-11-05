import re
from typing import Tuple, Optional, Dict, Any, List
from services.logger_setup import get_core_logger

logger = get_core_logger(__name__)

# Fast-path deterministic merchant classification patterns
MERCHANT_RULES = [
    # 1. Supermarkets & Groceries -> Food & Dining (Groceries)
    (
        r"(?i)\b(tesco|sainsburys?|lidl|morrisons?|co-?op|asda|waitrose|aldi|m&s|marks\s*&\s*spencer|iceland|spar|whole\s*foods|worldwide\s*foods|grocer|market|supermarket)\b",
        "Food & Dining",
        "Groceries"
    ),
    # 2. Restaurants, Cafes, Bakeries, Fast Food -> Food & Dining
    (
        r"(?i)\b(greggs|pret|costa|starbucks|subway|mcdonalds?|kfc|burger\s*king|nandos|wagamama|pizza|osteria|restaurant|cafe|caff|coffee|bakery|backwerk|bistro|diner|bar|pub|chillionrice|brickhouse|upper\s*crust)\b",
        "Food & Dining",
        "Restaurants & Cafes"
    ),
    # 3. Public Transit, Rail, Airlines -> Transportation (Public Transit & Rail)
    (
        r"(?i)\b(trainline|national\s*express|stagecoach|bee\s*network|tfl|transport\s*for\s*london|wiener\s*linien|u-bahn|s-bahn|öbb|obb|east\s*midlands\s*trains|northern\s*rail|lner|avanti|great\s*western|scotrail|railway|metro|bus|tram|flight|ryanair|easyjet|british\s*airways|menzies\s*aviation)\b",
        "Transportation",
        "Public Transit & Rail"
    ),
    # 4. Taxis & Rideshare -> Transportation (Taxis & Rideshare)
    (
        r"(?i)\b(uber|bolt|freenow|lyft|cab|taxi|rideshare)\b",
        "Transportation",
        "Taxis & Rideshare"
    ),
    # 5. Vehicle Fuel & Petrol -> Transportation (Vehicle Fuel / Petrol)
    (
        r"(?i)\b(shell|bp|esso|texaco|petrol|fuel|pfs)\b",
        "Transportation",
        "Vehicle Fuel / Petrol"
    ),
    # 6. Student Housing & Rent -> Housing (Rent)
    (
        r"(?i)\b(unite\s*students|student\s*roost|iq\s*student|student\s*castle|accommodation|landlord|letting|rent|residential|oead-wohnraum)\b",
        "Housing",
        "Rent"
    ),
    # 7. Utilities & Telecommunications -> Subscriptions & Digital Services / Utilities
    (
        r"(?i)\b(lebara|vodafone|ee|o2|three|giffgaff|bt|virgin\s*media|talktalk|a1\s*shop|telecom|mobile)\b",
        "Subscriptions & Digital Services",
        "Mobile & Broadband"
    ),
    # 8. Cloud, Software, Hosting -> Subscriptions & Digital Services (Software & Cloud Services)
    (
        r"(?i)\b(aws|amazon\s*web\s*services|google\s*cloud|azure|github|digitalocean|cloudflare|openai|anthropic|jetbrains|microsoft|adobe|slack|atlassian)\b",
        "Subscriptions & Digital Services",
        "Software & Cloud Services"
    ),
    # 9. Streaming Media -> Subscriptions & Digital Services (Streaming Video & Audio)
    (
        r"(?i)\b(netflix|spotify|disney\+?|youtube|apple\s*music|prime\s*video|audible|hulu|paramount|soundcloud)\b",
        "Subscriptions & Digital Services",
        "Streaming Video & Audio"
    ),
    # 10. Cinema, Events, Sightseeing -> Entertainment & Lifestyle (Events, Cinema & Theater)
    (
        r"(?i)\b(vue|odeon|cineworld|picturehouse|cinema|theater|theatre|ticketmaster|eventim|schloss\s*schoenbrunn|museum|palace|gallery)\b",
        "Entertainment & Lifestyle",
        "Events, Cinema & Theater"
    ),
    # 11. Government, Taxes, Visas, NHS -> Taxes & Government Payments / Healthcare
    (
        r"(?i)\b(ukvi|visa|ihs|immigration|hmrc|council\s*tax|dvla|gov\.uk)\b",
        "Taxes & Government Payments",
        "Income Tax / HMRC"
    ),
    # 12. Pharmacy & Healthcare -> Healthcare (Pharmacy & Medications)
    (
        r"(?i)\b(boots|superdrug|pharmacy|chemist|nhs|hospital|clinic|optician|specsavers|dental|dentist)\b",
        "Healthcare",
        "Pharmacy & Medications"
    ),
    # 13. Peer-to-Peer & Transfers -> Transfers & Payments (Peer-to-Peer Transfer)
    (
        r"(?i)\b(sent\s*money|received\s*money|splitwise|monzo\s*transfer|revolut\s*transfer|p2p|wire\s*transfer)\b",
        "Transfers & Payments",
        "Peer-to-Peer Transfer"
    ),
    # 14. Bank Fees & Currency Conversion -> Fees & Charges (Bank & Overdraft Fees)
    (
        r"(?i)\b(wise\s*charges|bank\s*fee|overdraft\s*fee|card\s*fee|fx\s*fee|international\s*transaction\s*fee)\b",
        "Fees & Charges",
        "Bank & Overdraft Fees"
    ),
    # 15. Real Clothing & Apparel -> Shopping & Retail (Clothing & Apparel)
    (
        r"(?i)\b(adidas|nike|zara|h&m|uniqlo|primark|next|asos|tk\s*maxx|pull\s*&\s*bear|bershka|apparel|clothing)\b",
        "Shopping & Retail",
        "Clothing & Apparel"
    ),
    # 16. Electronics -> Shopping & Retail (Electronics & Gadgets)
    (
        r"(?i)\b(currys|apple\s*store|samsung|bose|sony|argos)\b",
        "Shopping & Retail",
        "Electronics & Gadgets"
    ),
]

def classify_merchant(merchant_name: str, raw_description: str = "", amount: float = 0.0) -> Tuple[str, Optional[str]]:
    """
    Classifies a merchant name and raw transaction description into canonical
    (category, sub_category) without blind fallbacks.
    """
    text_to_test = f"{merchant_name} {raw_description}".strip()

    # Priority 1: Check P2P / Incoming money
    if re.search(r"(?i)^(received\s*money|transfer\s*from|from\s*papa|splitwise)", text_to_test) and amount > 0:
        return "Income", "Transfers & Gifts"
    if re.search(r"(?i)^(sent\s*money|transfer\s*to)", text_to_test) and amount < 0:
        return "Transfers & Payments", "Peer-to-Peer Transfer"

    # Priority 2: Check deterministic merchant regex rules
    for pattern, cat, sub_cat in MERCHANT_RULES:
        if re.search(pattern, text_to_test):
            return cat, sub_cat

    # If completely unmatched, return Uncategorized
    return "Uncategorized", None
