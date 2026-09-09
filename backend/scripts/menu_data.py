"""Curated menu catalogue used by the seed script.

Real dish names, prices in BDT, and hand-set attributes. This is data, not logic:
``seed.py`` draws from these pools to build each restaurant's menu. Dishes
deliberately repeat across restaurants (Kacchi Biryani is sold in several places
at different prices), which is both realistic and gives the collaborative
filtering more overlap to work with.
"""

from dataclasses import dataclass


@dataclass(frozen=True)
class DishSpec:
    """A menu dish template.

    ``base_price`` is in BDT before the per-restaurant price multiplier.
    """

    name: str
    cuisine: str
    spice_level: int
    is_veg: bool
    is_rice_based: bool
    base_price: int
    prep_time_min: int
    ingredient_tags: tuple[str, ...]


@dataclass(frozen=True)
class RestaurantSpec:
    name: str
    area: str
    # The cuisines the restaurant is actually known for; most of its menu.
    cuisine_tags: tuple[str, ...]
    # Multiplies every dish price; a Gulshan fine-dining place charges more for
    # the same plate than a Mirpur kabab shop.
    price_multiplier: float
    menu_size: int
    # How many menu slots go to desserts and drinks. Every restaurant sells
    # them, and spreading them across all venues matters for the recommender:
    # if desserts lived at one restaurant only, the dessert-lover cluster would
    # be separable by venue rather than by taste, which is not the structure we
    # want the collaborative filtering to have to find.
    side_quota: int


# ---------------------------------------------------------------------------
# Cuisine groups. The three seeded taste clusters are defined over these.
# ---------------------------------------------------------------------------
SPICY_DESI_CUISINES = ("bengali", "mughlai", "thai")
CONTINENTAL_CUISINES = ("italian", "continental", "fast_food", "chinese")
DESSERT_CUISINES = ("dessert", "beverage")

ALL_CUISINES = SPICY_DESI_CUISINES + CONTINENTAL_CUISINES + DESSERT_CUISINES

DHAKA_AREAS = (
    "Dhanmondi",
    "Gulshan",
    "Banani",
    "Uttara",
    "Mirpur",
    "Bashundhara",
    "Mohammadpur",
    "Old Dhaka",
)


# ---------------------------------------------------------------------------
# Dish catalogue
# ---------------------------------------------------------------------------
DISHES: tuple[DishSpec, ...] = (
    # ---- Bengali ----------------------------------------------------------
    DishSpec("Shorshe Ilish", "bengali", 4, False, False, 420, 30, ("fish", "mustard", "chilli")),
    DishSpec("Ilish Bhaja", "bengali", 2, False, False, 380, 20, ("fish", "oil")),
    DishSpec("Chingri Malai Curry", "bengali", 3, False, False, 450, 30, ("prawn", "coconut")),
    DishSpec("Beef Bhuna", "bengali", 4, False, False, 320, 35, ("beef", "onion", "chilli")),
    DishSpec("Chui Jhal Beef", "bengali", 5, False, False, 380, 40, ("beef", "chui", "chilli")),
    DishSpec("Kala Bhuna", "bengali", 5, False, False, 360, 40, ("beef", "chilli", "spice")),
    DishSpec("Rui Macher Jhol", "bengali", 3, False, False, 260, 25, ("fish", "potato")),
    DishSpec("Katla Kalia", "bengali", 3, False, False, 290, 28, ("fish", "onion", "yogurt")),
    DishSpec("Shutki Bhorta", "bengali", 5, False, False, 180, 15, ("fish", "chilli", "onion")),
    DishSpec("Aloo Bhorta", "bengali", 3, True, False, 90, 10, ("potato", "chilli", "onion")),
    DishSpec("Begun Bhaja", "bengali", 1, True, False, 100, 12, ("eggplant", "oil")),
    DishSpec("Dal Bhuna", "bengali", 2, True, False, 120, 18, ("lentil", "onion")),
    DishSpec("Murgir Jhol", "bengali", 3, False, False, 240, 25, ("chicken", "potato")),
    DishSpec("Bhuna Khichuri", "bengali", 3, False, True, 260, 30, ("rice", "lentil", "beef")),
    DishSpec("Sabji Khichuri", "bengali", 2, True, True, 180, 25, ("rice", "lentil", "vegetable")),
    DishSpec("Dim Bhuna", "bengali", 3, False, False, 160, 18, ("egg", "onion", "chilli")),
    DishSpec("Labra", "bengali", 2, True, False, 140, 20, ("vegetable", "lentil")),
    DishSpec("Shukto", "bengali", 1, True, False, 150, 22, ("vegetable", "milk")),
    DishSpec("Panta Bhat Platter", "bengali", 4, True, True, 200, 15, ("rice", "chilli", "onion")),
    DishSpec(
        "Chicken Rezala", "bengali", 3, False, False, 300, 32, ("chicken", "yogurt", "cashew")
    ),
    DishSpec("Loitta Shutki Bhuna", "bengali", 5, False, False, 220, 25, ("fish", "chilli")),
    DishSpec("Morog Polao", "bengali", 2, False, True, 320, 35, ("rice", "chicken", "ghee")),
    # ---- Mughlai ----------------------------------------------------------
    DishSpec(
        "Mutton Kacchi Biryani", "mughlai", 4, False, True, 520, 45, ("rice", "mutton", "ghee")
    ),
    DishSpec(
        "Chicken Kacchi Biryani", "mughlai", 4, False, True, 420, 45, ("rice", "chicken", "ghee")
    ),
    DishSpec("Beef Tehari", "mughlai", 4, False, True, 300, 35, ("rice", "beef", "mustard")),
    DishSpec("Mutton Tehari", "mughlai", 4, False, True, 360, 38, ("rice", "mutton")),
    DishSpec("Chicken Biryani", "mughlai", 3, False, True, 300, 35, ("rice", "chicken")),
    DishSpec("Beef Kala Bhuna", "mughlai", 5, False, False, 380, 40, ("beef", "chilli")),
    DishSpec("Chicken Chaap", "mughlai", 3, False, False, 260, 28, ("chicken", "yogurt", "spice")),
    DishSpec("Mutton Rezala", "mughlai", 3, False, False, 420, 38, ("mutton", "yogurt", "cashew")),
    DishSpec("Shami Kabab", "mughlai", 3, False, False, 180, 20, ("beef", "lentil", "spice")),
    DishSpec("Seekh Kabab", "mughlai", 4, False, False, 220, 22, ("beef", "chilli", "onion")),
    DishSpec("Chicken Tikka", "mughlai", 3, False, False, 240, 25, ("chicken", "yogurt", "spice")),
    DishSpec("Boti Kabab", "mughlai", 4, False, False, 260, 25, ("beef", "spice")),
    DishSpec("Beef Nehari", "mughlai", 4, False, False, 340, 50, ("beef", "spice", "bone")),
    DishSpec("Special Haleem", "mughlai", 4, False, False, 280, 45, ("lentil", "beef", "wheat")),
    DishSpec("Butter Naan", "mughlai", 0, True, False, 60, 10, ("flour", "butter")),
    DishSpec("Garlic Naan", "mughlai", 1, True, False, 70, 10, ("flour", "garlic")),
    DishSpec("Tandoori Roti", "mughlai", 0, True, False, 30, 8, ("flour",)),
    DishSpec("Mutton Leg Roast", "mughlai", 3, False, False, 620, 55, ("mutton", "spice")),
    DishSpec("Chicken Roast", "mughlai", 2, False, False, 240, 30, ("chicken", "ghee", "spice")),
    DishSpec("Firni", "mughlai", 0, True, True, 90, 15, ("rice", "milk", "sugar")),
    # ---- Thai -------------------------------------------------------------
    DishSpec("Tom Yum Soup", "thai", 4, False, False, 320, 25, ("prawn", "lemongrass", "chilli")),
    DishSpec(
        "Thai Green Curry", "thai", 4, False, False, 380, 30, ("chicken", "coconut", "chilli")
    ),
    DishSpec("Thai Red Curry", "thai", 5, False, False, 390, 30, ("beef", "coconut", "chilli")),
    DishSpec("Pad Thai Noodles", "thai", 3, False, False, 340, 25, ("noodles", "prawn", "peanut")),
    DishSpec("Thai Fried Rice", "thai", 3, False, True, 300, 22, ("rice", "chicken", "basil")),
    DishSpec("Basil Chicken", "thai", 4, False, False, 350, 25, ("chicken", "basil", "chilli")),
    DishSpec("Thai Clear Soup", "thai", 2, True, False, 220, 20, ("vegetable", "lemongrass")),
    DishSpec("Thai Papaya Salad", "thai", 4, True, False, 240, 15, ("papaya", "chilli", "peanut")),
    # ---- Chinese ----------------------------------------------------------
    DishSpec("Chicken Fried Rice", "chinese", 1, False, True, 260, 20, ("rice", "chicken", "egg")),
    DishSpec("Prawn Fried Rice", "chinese", 1, False, True, 340, 22, ("rice", "prawn", "egg")),
    DishSpec("Vegetable Fried Rice", "chinese", 1, True, True, 200, 18, ("rice", "vegetable")),
    DishSpec("Beef Chilli Onion", "chinese", 4, False, False, 340, 25, ("beef", "chilli", "onion")),
    DishSpec("Chicken Manchurian", "chinese", 3, False, False, 300, 25, ("chicken", "sauce")),
    DishSpec("Hot and Sour Soup", "chinese", 3, False, False, 200, 18, ("chicken", "vinegar")),
    DishSpec("Chicken Corn Soup", "chinese", 1, False, False, 190, 18, ("chicken", "corn", "egg")),
    DishSpec("Chicken Chowmein", "chinese", 2, False, False, 260, 20, ("noodles", "chicken")),
    DishSpec("Vegetable Chowmein", "chinese", 1, True, False, 200, 18, ("noodles", "vegetable")),
    DishSpec(
        "Sweet and Sour Chicken", "chinese", 1, False, False, 310, 25, ("chicken", "pineapple")
    ),
    DishSpec("Crispy Fried Chicken", "chinese", 2, False, False, 290, 25, ("chicken", "flour")),
    DishSpec("Szechuan Beef", "chinese", 5, False, False, 380, 28, ("beef", "chilli", "pepper")),
    DishSpec("Vegetable Spring Roll", "chinese", 1, True, False, 150, 15, ("vegetable", "flour")),
    DishSpec("Chicken Dumplings", "chinese", 1, False, False, 240, 22, ("chicken", "flour")),
    DishSpec("Chilli Prawn", "chinese", 4, False, False, 420, 28, ("prawn", "chilli")),
    DishSpec("Garlic Chicken", "chinese", 2, False, False, 300, 25, ("chicken", "garlic")),
    # ---- Italian ----------------------------------------------------------
    DishSpec("Margherita Pizza", "italian", 0, True, False, 480, 25, ("cheese", "tomato", "flour")),
    DishSpec("Pepperoni Pizza", "italian", 1, False, False, 620, 25, ("cheese", "beef", "flour")),
    DishSpec(
        "BBQ Chicken Pizza", "italian", 1, False, False, 640, 28, ("cheese", "chicken", "bbq")
    ),
    DishSpec("Four Cheese Pizza", "italian", 0, True, False, 700, 28, ("cheese", "flour")),
    DishSpec("Spicy Beef Pizza", "italian", 3, False, False, 680, 28, ("cheese", "beef", "chilli")),
    DishSpec(
        "Spaghetti Bolognese", "italian", 1, False, False, 460, 25, ("pasta", "beef", "tomato")
    ),
    DishSpec(
        "Fettuccine Alfredo", "italian", 0, True, False, 480, 25, ("pasta", "cheese", "cream")
    ),
    DishSpec("Penne Arrabbiata", "italian", 3, True, False, 420, 22, ("pasta", "tomato", "chilli")),
    DishSpec(
        "Chicken Lasagna", "italian", 1, False, False, 540, 35, ("pasta", "chicken", "cheese")
    ),
    DishSpec("Beef Lasagna", "italian", 1, False, False, 580, 35, ("pasta", "beef", "cheese")),
    DishSpec(
        "Chicken Parmigiana", "italian", 1, False, False, 560, 30, ("chicken", "cheese", "tomato")
    ),
    DishSpec("Mushroom Risotto", "italian", 0, True, True, 500, 30, ("rice", "mushroom", "cheese")),
    DishSpec("Bruschetta", "italian", 0, True, False, 260, 12, ("bread", "tomato", "basil")),
    DishSpec("Garlic Bread with Cheese", "italian", 0, True, False, 220, 12, ("bread", "cheese")),
    DishSpec("Caprese Salad", "italian", 0, True, False, 320, 10, ("cheese", "tomato", "basil")),
    DishSpec("Seafood Pasta", "italian", 1, False, False, 640, 30, ("pasta", "prawn", "cream")),
    # ---- Continental ------------------------------------------------------
    DishSpec("Grilled Chicken Steak", "continental", 1, False, False, 520, 30, ("chicken", "herb")),
    DishSpec("Beef Tenderloin Steak", "continental", 1, False, False, 880, 35, ("beef", "pepper")),
    DishSpec("Chicken Cordon Bleu", "continental", 0, False, False, 620, 35, ("chicken", "cheese")),
    DishSpec("Fish and Chips", "continental", 0, False, False, 480, 25, ("fish", "potato")),
    DishSpec("Roast Lamb Chops", "continental", 2, False, False, 920, 40, ("mutton", "herb")),
    DishSpec("Mashed Potato", "continental", 0, True, False, 180, 15, ("potato", "butter", "milk")),
    DishSpec(
        "Caesar Salad", "continental", 0, False, False, 340, 12, ("lettuce", "cheese", "chicken")
    ),
    DishSpec("Greek Salad", "continental", 0, True, False, 320, 10, ("cheese", "olive", "tomato")),
    DishSpec("Club Sandwich", "continental", 1, False, False, 360, 18, ("bread", "chicken", "egg")),
    DishSpec(
        "Cream of Mushroom Soup", "continental", 0, True, False, 240, 20, ("mushroom", "cream")
    ),
    DishSpec("Pumpkin Soup", "continental", 0, True, False, 220, 20, ("pumpkin", "cream")),
    DishSpec(
        "Grilled Fish Fillet", "continental", 1, False, False, 560, 28, ("fish", "lemon", "herb")
    ),
    DishSpec(
        "Chicken Stroganoff", "continental", 1, False, True, 540, 30, ("chicken", "cream", "rice")
    ),
    DishSpec(
        "Beef Steak Pepper Sauce", "continental", 2, False, False, 840, 35, ("beef", "pepper")
    ),
    DishSpec("Herb Roast Chicken", "continental", 1, False, False, 580, 40, ("chicken", "herb")),
    DishSpec("Potato Wedges", "continental", 1, True, False, 200, 15, ("potato", "spice")),
    # ---- Fast food --------------------------------------------------------
    DishSpec("Beef Burger", "fast_food", 1, False, False, 280, 15, ("bread", "beef", "cheese")),
    DishSpec("Cheese Burger", "fast_food", 1, False, False, 320, 15, ("bread", "beef", "cheese")),
    DishSpec("Chicken Burger", "fast_food", 1, False, False, 240, 15, ("bread", "chicken")),
    DishSpec(
        "Zinger Burger", "fast_food", 3, False, False, 300, 18, ("bread", "chicken", "chilli")
    ),
    DishSpec("Veggie Burger", "fast_food", 1, True, False, 200, 15, ("bread", "vegetable")),
    DishSpec("French Fries", "fast_food", 0, True, False, 130, 10, ("potato", "salt")),
    DishSpec("Cheese Loaded Fries", "fast_food", 1, True, False, 220, 12, ("potato", "cheese")),
    DishSpec("Spicy Chicken Wings", "fast_food", 4, False, False, 290, 20, ("chicken", "chilli")),
    DishSpec("BBQ Chicken Wings", "fast_food", 2, False, False, 290, 20, ("chicken", "bbq")),
    DishSpec("Chicken Nuggets", "fast_food", 1, False, False, 210, 12, ("chicken", "flour")),
    DishSpec("Beef Shawarma", "fast_food", 2, False, False, 260, 15, ("bread", "beef", "garlic")),
    DishSpec("Chicken Shawarma", "fast_food", 2, False, False, 220, 15, ("bread", "chicken")),
    DishSpec("Chicken Wrap", "fast_food", 2, False, False, 250, 15, ("bread", "chicken", "sauce")),
    DishSpec("Loaded Nachos", "fast_food", 2, False, False, 320, 15, ("corn", "cheese", "beef")),
    # ---- Dessert ----------------------------------------------------------
    DishSpec("Rasgulla", "dessert", 0, True, False, 70, 5, ("milk", "sugar")),
    DishSpec("Rasmalai", "dessert", 0, True, False, 110, 8, ("milk", "sugar", "cardamom")),
    DishSpec("Chomchom", "dessert", 0, True, False, 80, 5, ("milk", "sugar")),
    DishSpec("Mishti Doi", "dessert", 0, True, False, 90, 5, ("yogurt", "sugar")),
    DishSpec("Kalojam", "dessert", 0, True, False, 75, 5, ("milk", "sugar", "flour")),
    DishSpec("Sandesh", "dessert", 0, True, False, 85, 5, ("milk", "sugar")),
    DishSpec("Jilapi", "dessert", 0, True, False, 60, 8, ("flour", "sugar")),
    DishSpec("Shemai", "dessert", 0, True, False, 90, 15, ("milk", "sugar", "noodles")),
    DishSpec("Chaler Payesh", "dessert", 0, True, True, 110, 20, ("rice", "milk", "sugar")),
    DishSpec("Nuts Falooda", "dessert", 0, True, False, 220, 12, ("milk", "nuts", "jelly")),
    DishSpec(
        "New York Cheesecake", "dessert", 0, True, False, 320, 10, ("cheese", "sugar", "biscuit")
    ),
    DishSpec(
        "Chocolate Brownie", "dessert", 0, True, False, 240, 10, ("chocolate", "flour", "sugar")
    ),
    DishSpec("Red Velvet Cake", "dessert", 0, True, False, 280, 10, ("flour", "cheese", "sugar")),
    DishSpec("Tiramisu", "dessert", 0, True, False, 340, 10, ("coffee", "cheese", "cocoa")),
    DishSpec("Chocolate Lava Cake", "dessert", 0, True, False, 300, 15, ("chocolate", "flour")),
    DishSpec("Ice Cream Sundae", "dessert", 0, True, False, 200, 8, ("milk", "chocolate", "nuts")),
    DishSpec("Vanilla Custard", "dessert", 0, True, False, 130, 10, ("milk", "egg", "sugar")),
    DishSpec("Caramel Pudding", "dessert", 0, True, False, 160, 10, ("milk", "egg", "sugar")),
    DishSpec("Gulab Jamun", "dessert", 0, True, False, 70, 5, ("milk", "sugar", "flour")),
    DishSpec("Mango Cheesecake", "dessert", 0, True, False, 340, 10, ("mango", "cheese", "sugar")),
    # ---- Beverage ---------------------------------------------------------
    DishSpec("Masala Cha", "beverage", 2, True, False, 50, 6, ("tea", "milk", "spice")),
    DishSpec("Doodh Cha", "beverage", 0, True, False, 40, 5, ("tea", "milk", "sugar")),
    DishSpec("Lemon Tea", "beverage", 0, True, False, 40, 5, ("tea", "lemon")),
    DishSpec("Espresso", "beverage", 0, True, False, 150, 5, ("coffee",)),
    DishSpec("Cappuccino", "beverage", 0, True, False, 220, 8, ("coffee", "milk")),
    DishSpec("Cafe Latte", "beverage", 0, True, False, 230, 8, ("coffee", "milk")),
    DishSpec("Cold Coffee", "beverage", 0, True, False, 250, 8, ("coffee", "milk", "ice")),
    DishSpec("Mango Lassi", "beverage", 0, True, False, 180, 8, ("mango", "yogurt")),
    DishSpec("Borhani", "beverage", 3, True, False, 90, 6, ("yogurt", "mint", "chilli")),
    DishSpec("Fresh Lemonade", "beverage", 0, True, False, 110, 6, ("lemon", "sugar")),
    DishSpec("Faluda Shake", "beverage", 0, True, False, 240, 10, ("milk", "nuts", "jelly")),
    DishSpec("Chocolate Shake", "beverage", 0, True, False, 260, 8, ("milk", "chocolate")),
    DishSpec("Strawberry Shake", "beverage", 0, True, False, 260, 8, ("milk", "strawberry")),
    DishSpec("Sugarcane Juice", "beverage", 0, True, False, 80, 5, ("sugarcane", "ice")),
)


SIDE_CUISINES = ("dessert", "beverage")

RESTAURANTS: tuple[RestaurantSpec, ...] = (
    RestaurantSpec("Kacchi Bhai", "Dhanmondi", ("mughlai", "bengali"), 1.00, 40, 8),
    RestaurantSpec("Sultan's Dine", "Gulshan", ("mughlai", "bengali"), 1.25, 38, 8),
    RestaurantSpec("Star Kabab and Restaurant", "Mirpur", ("mughlai", "bengali"), 0.85, 42, 8),
    RestaurantSpec("Bhoj Bangla Kitchen", "Old Dhaka", ("bengali", "mughlai"), 0.90, 40, 8),
    RestaurantSpec("Pizza Roma", "Banani", ("italian", "fast_food"), 1.15, 36, 10),
    RestaurantSpec("The Continental Grill", "Gulshan", ("continental", "italian"), 1.35, 38, 10),
    RestaurantSpec("Dragon Wok", "Uttara", ("chinese", "thai"), 1.05, 34, 12),
    # Already a dessert house, so no separate side quota.
    RestaurantSpec("Sweet Bengal Cafe", "Dhanmondi", ("dessert", "beverage"), 1.00, 34, 0),
)

TARGET_ITEM_COUNT = sum(r.menu_size for r in RESTAURANTS)
