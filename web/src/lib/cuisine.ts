/**
 * Visual identity per cuisine.
 *
 * The seeded catalogue points `image_url` at files that do not exist yet, so
 * every dish needs a placeholder that looks deliberate rather than broken. A
 * cuisine-tinted gradient plus a dish glyph reads as design; a grey box with a
 * torn-image icon reads as a bug. `DishImage` still prefers a real photo when
 * one is present, so dropping assets into `public/images/food/` upgrades the
 * whole site with no code change.
 */

export interface CuisineStyle {
  /** Tailwind gradient classes, still used for the decorative hero grid. */
  gradient: string;
  /** Base hue for dish tiles; individual dishes sit within +/- HUE_SPREAD of it. */
  hue: number;
  /** Badge colours for cuisine chips. */
  badge: string;
  glyph: string;
  label: string;
}

const FALLBACK: CuisineStyle = {
  hue: 30,
  gradient: "from-stone-300 to-stone-400 dark:from-stone-700 dark:to-stone-800",
  badge:
    "bg-stone-100 text-stone-700 dark:bg-stone-800 dark:text-stone-300",
  glyph: "🍽️",
  label: "Other",
};

export const CUISINE_STYLES: Record<string, CuisineStyle> = {
  bengali: {
    hue: 38,
    gradient: "from-amber-300 to-orange-500 dark:from-amber-600 dark:to-orange-800",
    badge: "bg-amber-100 text-amber-800 dark:bg-amber-950 dark:text-amber-300",
    glyph: "🐟",
    label: "Bengali",
  },
  mughlai: {
    hue: 20,
    gradient: "from-orange-400 to-red-600 dark:from-orange-700 dark:to-red-900",
    badge: "bg-orange-100 text-orange-800 dark:bg-orange-950 dark:text-orange-300",
    glyph: "🍛",
    label: "Mughlai",
  },
  chinese: {
    hue: 2,
    gradient: "from-red-300 to-rose-500 dark:from-red-700 dark:to-rose-900",
    badge: "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300",
    glyph: "🥡",
    label: "Chinese",
  },
  thai: {
    hue: 158,
    gradient: "from-emerald-300 to-teal-600 dark:from-emerald-700 dark:to-teal-900",
    badge: "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300",
    glyph: "🍜",
    label: "Thai",
  },
  italian: {
    hue: 340,
    gradient: "from-rose-300 to-red-500 dark:from-rose-700 dark:to-red-900",
    badge: "bg-rose-100 text-rose-800 dark:bg-rose-950 dark:text-rose-300",
    glyph: "🍕",
    label: "Italian",
  },
  continental: {
    hue: 215,
    gradient: "from-slate-300 to-slate-500 dark:from-slate-600 dark:to-slate-800",
    badge: "bg-slate-100 text-slate-700 dark:bg-slate-800 dark:text-slate-300",
    glyph: "🥩",
    label: "Continental",
  },
  fast_food: {
    hue: 44,
    gradient: "from-yellow-300 to-amber-500 dark:from-yellow-700 dark:to-amber-900",
    badge: "bg-yellow-100 text-yellow-800 dark:bg-yellow-950 dark:text-yellow-300",
    glyph: "🍔",
    label: "Fast Food",
  },
  dessert: {
    hue: 315,
    gradient: "from-pink-300 to-fuchsia-500 dark:from-pink-700 dark:to-fuchsia-900",
    badge: "bg-pink-100 text-pink-800 dark:bg-pink-950 dark:text-pink-300",
    glyph: "🍰",
    label: "Dessert",
  },
  beverage: {
    hue: 195,
    gradient: "from-sky-300 to-cyan-500 dark:from-sky-700 dark:to-cyan-900",
    badge: "bg-sky-100 text-sky-800 dark:bg-sky-950 dark:text-sky-300",
    glyph: "☕",
    label: "Beverage",
  },
};

export function cuisineStyle(cuisine: string): CuisineStyle {
  return CUISINE_STYLES[cuisine] ?? FALLBACK;
}

/**
 * Pick a glyph from the dish name where it is more specific than the cuisine's
 * default, so a menu of thirty Mughlai dishes is not thirty identical tiles.
 */
const NAME_GLYPHS: Array<[RegExp, string]> = [
  [/biryani|tehari|polao|khichuri|rice/i, "🍚"],
  [/kabab|tikka|seekh|boti|grill|tandoor/i, "🍢"],
  [/burger/i, "🍔"],
  [/pizza/i, "🍕"],
  [/pasta|spaghetti|lasagna|fettuccine|penne/i, "🍝"],
  [/noodle|chowmein|pad thai/i, "🍜"],
  [/soup/i, "🥣"],
  [/salad/i, "🥗"],
  [/ilish|macher|fish|prawn|chingri/i, "🐟"],
  [/beef|mutton|steak|bhuna|nehari|rezala/i, "🥩"],
  [/chicken|murgir|roast/i, "🍗"],
  [/cake|brownie|tiramisu|cheesecake|pudding/i, "🍰"],
  [/rasgulla|rasmalai|jilapi|sandesh|jamun|payesh|firni|doi|chomchom|kalojam/i, "🍮"],
  [/ice cream|falooda|sundae/i, "🍨"],
  [/coffee|espresso|cappuccino|latte/i, "☕"],
  [/tea|cha/i, "🍵"],
  [/shake|lassi|juice|lemonade|borhani/i, "🥤"],
  [/naan|roti|bread|paratha/i, "🫓"],
  [/fries|wedges|nachos/i, "🍟"],
  [/wings/i, "🍗"],
  [/shawarma|wrap/i, "🌯"],
  [/egg|dim/i, "🥚"],
  [/bhorta|bhaji|begun|labra|shukto|sabji|vegetable/i, "🥘"],
];

export function dishGlyph(name: string, cuisine: string): string {
  for (const [pattern, glyph] of NAME_GLYPHS) {
    if (pattern.test(name)) return glyph;
  }
  return cuisineStyle(cuisine).glyph;
}

/** How far either side of the cuisine's base hue a dish may sit.
 *
 * Kept narrow on purpose: at +/-11 the fast-food band reached hue 59, which is
 * chartreuse and reads as off food rather than as a warm yellow. */
const HUE_SPREAD = 7;

/**
 * A stable hue for one dish: its cuisine's base, nudged by a hash of the name.
 *
 * Deterministic, so a dish looks the same on every page and between reloads,
 * and bounded, so a Thai dish never drifts far enough to read as Italian.
 */
export function dishHue(name: string, cuisine: string): number {
  let hash = 0;
  for (let index = 0; index < name.length; index += 1) {
    hash = (hash * 31 + name.charCodeAt(index)) % 100000;
  }
  const offset = (hash % (HUE_SPREAD * 2 + 1)) - HUE_SPREAD;
  return (cuisineStyle(cuisine).hue + offset + 360) % 360;
}

export const SPICE_LABELS = [
  "No chilli",
  "Mild",
  "Lightly spiced",
  "Medium hot",
  "Hot",
  "Fiery",
];
