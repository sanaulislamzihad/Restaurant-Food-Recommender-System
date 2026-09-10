/**
 * Types mirroring the FastAPI response schemas.
 *
 * Two things the backend does that are easy to get wrong here:
 *
 * - Money is a `string`, not a number. Pydantic serialises `Decimal` as a
 *   decimal string so no precision is lost in transit. Parse it at the point of
 *   arithmetic with `parsePrice`, never store it as a float.
 * - Timestamps are UTC ISO strings ending in `Z`.
 */

export type Gender = "male" | "female" | "other";

export type OrderStatus =
  | "pending"
  | "confirmed"
  | "preparing"
  | "delivered"
  | "cancelled";

export type MenuSort =
  | "name"
  | "price_asc"
  | "price_desc"
  | "rating"
  | "popularity"
  | "newest";

export type RecommendationSource =
  | "similar_to_ordered"
  | "favourite_cuisine"
  | "popular"
  | "trending"
  | "collaborative";

export interface Page<T> {
  items: T[];
  total: number;
  limit: number;
  offset: number;
}

export interface RestaurantSummary {
  id: number;
  name: string;
  area: string;
  cuisine_tags: string[];
}

export interface FoodItemSummary {
  id: number;
  name: string;
  cuisine: string;
  /** Decimal string, e.g. "450.00". */
  price: string;
  spice_level: number;
  is_veg: boolean;
  is_rice_based: boolean;
  prep_time_min: number;
  image_url: string | null;
  is_available: boolean;
  restaurant_id: number;
}

export interface FoodItemDetail extends FoodItemSummary {
  description: string;
  ingredient_tags: string[];
  created_at: string;
  restaurant: RestaurantSummary | null;
  average_rating: number | null;
  rating_count: number;
}

export interface RecommendedItem {
  item: FoodItemSummary;
  score: number;
  reason: string;
  source: RecommendationSource;
}

export interface RecommendationResponse {
  items: RecommendedItem[];
  /** Labels both halves, e.g. "v1+v4"; null when the response is a fallback. */
  model_version: string | null;
  is_cold_start: boolean;
  candidates_considered: number;
  /** Pipeline diagnostics. Useful for the admin view and for debugging a feed. */
  retrieval_ms: number;
  ranking_ms: number;
  retrieval_sources: Record<string, number>;
  scoring_breakdown: Record<string, number>;
  dropped_recently_ordered: number;
  latency_ms: number;
  cached: boolean;
}

export interface TokenResponse {
  access_token: string;
  token_type: string;
  expires_in_minutes: number;
}

export interface UserResponse {
  id: number;
  name: string;
  email: string;
  age: number | null;
  gender: Gender | null;
  area: string | null;
  spice_tolerance: number;
  created_at: string;
}

export interface CuisineAffinity {
  cuisine: string;
  order_count: number;
  average_rating: number | null;
}

export interface TasteProfile {
  total_ratings: number;
  total_orders: number;
  average_rating_given: number | null;
  top_cuisines: CuisineAffinity[];
  is_cold_start: boolean;
}

export interface RatingResponse {
  id: number;
  user_id: number;
  food_item_id: number;
  /** Decimal string, e.g. "4.0". */
  rating: string;
  created_at: string;
}

export interface RatingWithItem extends RatingResponse {
  food_item: FoodItemSummary;
}

export interface OrderItemResponse {
  id: number;
  food_item_id: number;
  quantity: number;
  unit_price: string;
  food_item: FoodItemSummary | null;
}

export interface OrderResponse {
  id: number;
  user_id: number;
  total_amount: string;
  status: OrderStatus;
  created_at: string;
  items: OrderItemResponse[];
}

export interface ImpressionBatchResponse {
  recorded: number;
  skipped: number;
}

export interface RegisterPayload {
  name: string;
  email: string;
  password: string;
  age?: number | null;
  gender?: Gender | null;
  area?: string | null;
  spice_tolerance?: number;
}

export interface MenuQuery {
  cuisine?: string[];
  min_price?: number;
  max_price?: number;
  is_veg?: boolean;
  max_spice?: number;
  restaurant_id?: number;
  search?: string;
  available_only?: boolean;
  sort?: MenuSort;
  limit?: number;
  offset?: number;
}
