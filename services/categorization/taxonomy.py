from typing import Dict, List, Optional, Tuple, Set

# The 13 canonical top-level BudAI categories
STANDARD_CATEGORIES: List[str] = [
    "Income",
    "Housing",
    "Food & Dining",
    "Transportation",
    "Utilities",
    "Entertainment & Lifestyle",
    "Subscriptions & Digital Services",
    "Shopping & Retail",
    "Healthcare",
    "Transfers & Payments",
    "Fees & Charges",
    "Savings & Investments",
    "Taxes & Government Payments"
]

# Taxonomy mapping: (Category, Sub-Category) -> {ons_code, default_tags, is_essential}
TAXONOMY_MAP: Dict[str, Dict[str, Dict[str, any]]] = {
    "Food & Dining": {
        "Groceries": {
            "ons_code": "CP01",
            "ons_name": "Food and non-alcoholic beverages",
            "tags": ["#essential", "#ons:cp01"],
            "staple_keywords": {
                "milk": "CZOH",
                "bread": "CZNT",
                "eggs": "CZOM",
                "butter": "CZOL"
            }
        },
        "Restaurants & Cafes": {
            "ons_code": "CP11",
            "ons_name": "Restaurants and hotels",
            "tags": ["#discretionary", "#ons:cp11"]
        },
        "Takeaway & Fast Food": {
            "ons_code": "CP11",
            "ons_name": "Restaurants and hotels",
            "tags": ["#discretionary", "#ons:cp11"]
        },
        "Bars & Pubs": {
            "ons_code": "CP11",
            "ons_name": "Restaurants and hotels",
            "tags": ["#discretionary", "#ons:cp11"]
        }
    },
    "Utilities": {
        "Electricity & Gas": {
            "ons_code": "CP04_ENERGY",
            "ons_name": "Electricity, gas and other fuels",
            "tags": ["#recurring", "#essential", "#ons:cp04_energy"]
        },
        "Water & Sewage": {
            "ons_code": "CP04_WATER",
            "ons_name": "Water supply and miscellaneous services",
            "tags": ["#recurring", "#essential", "#ons:cp04_water"]
        },
        "Waste & Council Tax": {
            "ons_code": "CP04_COUNCIL",
            "ons_name": "Council tax and rates",
            "tags": ["#recurring", "#fixed", "#ons:cp04_council"]
        }
    },
    "Transportation": {
        "Vehicle Fuel / Petrol": {
            "ons_code": "CP07_FUEL",
            "ons_name": "Fuels and lubricants",
            "tags": ["#essential", "#ons:cp07_fuel"],
            "staple_keywords": {
                "petrol": "CZPC",
                "diesel": "CZPE"
            }
        },
        "Public Transit & Rail": {
            "ons_code": "CP07_TRANSIT",
            "ons_name": "Passenger transport services",
            "tags": ["#ons:cp07_transit"]
        },
        "Taxis & Rideshare": {
            "ons_code": "CP07_TRANSIT",
            "ons_name": "Passenger transport services",
            "tags": ["#discretionary", "#ons:cp07_transit"]
        },
        "Vehicle Maintenance & Parking": {
            "ons_code": "CP07",
            "ons_name": "Transport",
            "tags": ["#ons:cp07"]
        }
    },
    "Housing": {
        "Rent": {
            "ons_code": "CP04_HOUSING",
            "ons_name": "Actual rentals for housing",
            "tags": ["#fixed", "#essential", "#recurring", "#ons:cp04_housing"]
        },
        "Mortgage Payment": {
            "ons_code": "CP04_HOUSING",
            "ons_name": "Owner occupiers housing costs",
            "tags": ["#fixed", "#essential", "#recurring", "#ons:cp04_housing"]
        },
        "Home Repairs & Maintenance": {
            "ons_code": "CP04",
            "ons_name": "Housing, water, electricity, gas and other fuels",
            "tags": ["#ons:cp04"]
        }
    },
    "Subscriptions & Digital Services": {
        "Streaming Video & Audio": {
            "ons_code": "CP09",
            "ons_name": "Recreation and culture",
            "tags": ["#recurring", "#discretionary", "#ons:cp09_digital"]
        },
        "Software & Cloud Services": {
            "ons_code": "CP09",
            "ons_name": "Recreation and culture",
            "tags": ["#recurring", "#ons:cp09_digital"]
        },
        "Mobile & Broadband": {
            "ons_code": "CP08",
            "ons_name": "Communication",
            "tags": ["#recurring", "#essential", "#ons:cp08_telecom"]
        }
    },
    "Entertainment & Lifestyle": {
        "Events, Cinema & Theater": {
            "ons_code": "CP09",
            "ons_name": "Recreation and culture",
            "tags": ["#discretionary", "#ons:cp09_leisure"]
        },
        "Gym & Fitness": {
            "ons_code": "CP09",
            "ons_name": "Recreation and culture",
            "tags": ["#recurring", "#ons:cp09_leisure"]
        },
        "Hobbies & Sports": {
            "ons_code": "CP09",
            "ons_name": "Recreation and culture",
            "tags": ["#discretionary", "#ons:cp09_leisure"]
        }
    },
    "Shopping & Retail": {
        "Clothing & Apparel": {
            "ons_code": "CP03",
            "ons_name": "Clothing and footwear",
            "tags": ["#discretionary", "#ons:cp03_apparel"]
        },
        "Electronics & Gadgets": {
            "ons_code": "CP09",
            "ons_name": "Recreation and culture",
            "tags": ["#discretionary", "#ons:cp09"]
        },
        "Furniture & Household": {
            "ons_code": "CP05",
            "ons_name": "Furnishings, household equipment and maintenance",
            "tags": ["#ons:cp05_household"]
        },
        "General Merchandise": {
            "ons_code": "CP00",
            "ons_name": "All items",
            "tags": ["#ons:cp00"]
        }
    },
    "Healthcare": {
        "Pharmacy & Medications": {
            "ons_code": "CP06",
            "ons_name": "Health",
            "tags": ["#essential", "#ons:cp06_health"]
        },
        "Dental & Optical": {
            "ons_code": "CP06",
            "ons_name": "Health",
            "tags": ["#essential", "#ons:cp06_health"]
        },
        "Medical Services": {
            "ons_code": "CP06",
            "ons_name": "Health",
            "tags": ["#essential", "#ons:cp06_health"]
        }
    },
    "Income": {
        "Salary & Wages": {
            "ons_code": "EARN01",
            "ons_name": "Average weekly earnings",
            "tags": ["#income", "#recurring", "#ons:earn01"]
        },
        "Bonus & Commission": {
            "ons_code": "EARN01",
            "ons_name": "Average weekly earnings",
            "tags": ["#income", "#ons:earn01"]
        },
        "Investments & Dividends": {
            "ons_code": None,
            "ons_name": None,
            "tags": ["#income", "#capital"]
        },
        "Refunds & Reimbursements": {
            "ons_code": None,
            "ons_name": None,
            "tags": ["#income"]
        }
    },
    "Transfers & Payments": {
        "Internal Account Transfer": {
            "ons_code": None,
            "ons_name": None,
            "tags": ["#transfer", "#internal"]
        },
        "Peer-to-Peer Transfer": {
            "ons_code": None,
            "ons_name": None,
            "tags": ["#transfer"]
        }
    },
    "Fees & Charges": {
        "Bank & Overdraft Fees": {
            "ons_code": "CP12",
            "ons_name": "Miscellaneous goods and services",
            "tags": ["#fee", "#penalty"]
        },
        "Interest Paid": {
            "ons_code": "CP12",
            "ons_name": "Miscellaneous goods and services",
            "tags": ["#fee", "#interest"]
        }
    },
    "Savings & Investments": {
        "Savings Deposit": {
            "ons_code": None,
            "ons_name": None,
            "tags": ["#savings", "#allocating"]
        },
        "Stock / ETF Purchase": {
            "ons_code": None,
            "ons_name": None,
            "tags": ["#investment", "#allocating"]
        },
        "Pension Contribution": {
            "ons_code": None,
            "ons_name": None,
            "tags": ["#pension", "#recurring"]
        }
    },
    "Taxes & Government Payments": {
        "Income Tax / HMRC": {
            "ons_code": None,
            "ons_name": None,
            "tags": ["#tax", "#statutory"]
        },
        "Fines & Penalties": {
            "ons_code": None,
            "ons_name": None,
            "tags": ["#penalty", "#government"]
        }
    }
}

def enrich_transaction_taxonomy(
    category: str,
    sub_category: Optional[str] = None,
    description: str = "",
    existing_tags: Optional[List[str]] = None
) -> Tuple[str, Optional[str], List[str], Optional[str]]:
    """
    Enriches a transaction with canonical classification, sub-category, 
    and ONS macro tags.

    Returns:
        (canonical_category, canonical_sub_category, enriched_tags_list, ons_code)
    """
    tags_set: Set[str] = set(existing_tags or [])

    # If category is empty or not in standard canonical categories, do NOT force Shopping & Retail
    if not category or category not in STANDARD_CATEGORIES:
        return "Uncategorized", None, sorted(list(tags_set)), None

    cat_match = category

    # Match or default sub-category within the valid category map
    sub_map = TAXONOMY_MAP.get(cat_match, {})
    sub_cat_match = sub_category if (sub_category and sub_category in sub_map) else (list(sub_map.keys())[0] if sub_map else None)

    ons_code = None
    if sub_cat_match and sub_cat_match in sub_map:
        spec = sub_map[sub_cat_match]
        default_tags = spec.get("tags", [])
        ons_code = spec.get("ons_code")

        for t in default_tags:
            tags_set.add(t)

        # Check for granular staple keyword matches in description
        staple_keywords = spec.get("staple_keywords", {})
        desc_lower = (description or "").lower()
        for kw, series_id in staple_keywords.items():
            if kw in desc_lower:
                tags_set.add(f"#staple:{kw}")
                tags_set.add(f"#ons:{series_id.lower()}")

    return cat_match, sub_cat_match, sorted(list(tags_set)), ons_code
