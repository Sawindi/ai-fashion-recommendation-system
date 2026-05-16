# src/events_config.py

from dataclasses import dataclass
from typing import List, Dict


@dataclass
class EventProfile:
    name: str
    # Outfit must satisfy at least one of these templates.
    required_any: List[List[str]]

    preferred_categories: List[str]
    forbidden_categories: List[str]
    formality_level: int  # 1 (casual) -> 5 (very formal)
    preferred_colours: List[str]


EVENTS: Dict[str, EventProfile] = {

    # 1. Casual Day Out (very flexible)
    "casual_day_out": EventProfile(
        name="casual_day_out",
        required_any=[
            ["top", "trousers"],
            ["top", "skirt"],
            ["top", "shorts"],
            ["t-shirt", "trousers"],
            ["t-shirt", "shorts"],
            ["dress"],
        ],
        preferred_categories=["t-shirt", "jeans", "cardigan", "skirt", "shorts"],
        forbidden_categories=["blazer"],
        formality_level=1,
        preferred_colours=[]
    ),

    # 2. Office Work (smart outfits, not very strict like interview)
    "office_work": EventProfile(
        name="office_work",
        required_any=[
            ["trousers", "shirt"],
            ["trousers", "top"],
            ["trousers", "knitwear"],
            ["trousers", "cardigan"],
            ["dress", "blazer"],
        ],
        preferred_categories=["blazer", "shirt", "coat", "trousers"],
        forbidden_categories=["shorts"],
        formality_level=4,
        preferred_colours=["black", "grey", "white"]
    ),

    # 3. University / Lecture (practical + casual-smart)
    "university": EventProfile(
        name="university",
        required_any=[
            ["top", "jeans"],
            ["top", "trousers"],
            ["t-shirt", "jeans"],
            ["sweatshirt", "jeans"],
            ["top", "skirt"],
        ],
        preferred_categories=["jeans", "cardigan", "knitwear", "sweatshirt"],
        forbidden_categories=["blazer"],
        formality_level=2,
        preferred_colours=[]
    ),

    # 4. Date Night
    "date_night": EventProfile(
        name="date_night",
        required_any=[
            ["dress"],
            ["top", "trousers"],
            ["top", "skirt"],
            ["shirt", "trousers"],
            ["knitwear", "skirt"],
        ],
        preferred_categories=["dress"],
        forbidden_categories=["sweatshirt"],
        formality_level=3,
        preferred_colours=["red", "pink"]
    ),

    # 5. Wedding Guest (mostly dress-based)
    "wedding_guest": EventProfile(
        name="wedding_guest",
        required_any=[
            ["dress"],            # accept dress
            ["dress", "blazer"],  # dress + blazer/coat
            ["dress", "coat"],
        ],
        preferred_categories=["blazer", "coat"],
        forbidden_categories=["shorts", "jeans", "sweatshirt"],
        formality_level=5,
        preferred_colours=["blue", "green", "pink", "red", "orange"]
    ),

    # 6. Job Interview
    "job_interview": EventProfile(
        name="job_interview",
        required_any=[
            ["blazer", "trousers"],  # must have both
        ],
        preferred_categories=["shirt", "coat"],
        forbidden_categories=["shorts", "jeans", "sweatshirt", "dress"],
        formality_level=5,
        preferred_colours=["black", "grey", "blue", "white"]
    ),

    # 7. Party 
    "party": EventProfile(
        name="party",
        required_any=[
            ["dress"],
            ["top", "skirt"],
            ["top", "trousers"],
        ],
        preferred_categories=["dress", "skirt", "jacket"],
        forbidden_categories=["shirt"],
        formality_level=3,
        preferred_colours=["black", "red", "purple", "pink"]
    ),

    # 8. Formal Dinner
    "formal_dinner": EventProfile(
        name="formal_dinner",
        required_any=[
            ["dress"],
            ["dress", "blazer"],
            ["dress", "coat"],
            ["shirt", "trousers", "blazer"],
            ["top", "trousers", "blazer"],
        ],
        preferred_categories=["blazer", "coat", "dress", "shirt"],
        forbidden_categories=["shorts", "jeans", "sweatshirt"],
        formality_level=4,
        preferred_colours=["black", "green", "blue", "red"]
    ),

    # 9. Travel / Airport (comfort)
    "travel": EventProfile(
        name="travel",
        required_any=[
            ["trousers", "top"],
            ["trousers", "sweatshirt"],
            ["jeans", "top"],
            ["jeans", "sweatshirt"],
        ],
        preferred_categories=["sweatshirt", "jacket", "top"],
        forbidden_categories=["blazer"],
        formality_level=1,
        preferred_colours=[]
    ),

    # 10. Sport / Gym (activewear)
    "sport_activity": EventProfile(
        name="sport_activity",
        required_any=[
            ["shorts", "top"],
            ["shorts", "sweatshirt"],
        ],
        preferred_categories=["sweatshirt", "top"],
        forbidden_categories=["blazer", "dress", "coat"],
        formality_level=1,
        preferred_colours=[]
    ),
}