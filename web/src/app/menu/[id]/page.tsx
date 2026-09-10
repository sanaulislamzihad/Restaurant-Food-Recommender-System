"use client";

import { useMutation, useQuery, useQueryClient } from "@tanstack/react-query";
import { ArrowLeft, Clock, Flame, Leaf, MapPin, Minus, Plus } from "lucide-react";
import Link from "next/link";
import { useParams } from "next/navigation";
import * as React from "react";

import { DishImage } from "@/components/dish-image";
import { DishRow } from "@/components/dish-row";
import { RatingInput, RatingStars } from "@/components/rating-stars";
import { Button } from "@/components/ui/button";
import { Alert, Badge, Card, Skeleton } from "@/components/ui/primitives";
import { ApiError, api } from "@/lib/api";
import { SPICE_LABELS, cuisineStyle } from "@/lib/cuisine";
import { formatPrice, humanise } from "@/lib/utils";
import { useAuth } from "@/providers/app-providers";
import { useCart } from "@/store/cart";

export default function ItemDetailPage() {
  const params = useParams<{ id: string }>();
  const itemId = Number(params.id);
  const { user } = useAuth();
  const queryClient = useQueryClient();
  const add = useCart((state) => state.add);

  const [quantity, setQuantity] = React.useState(1);
  const [myRating, setMyRating] = React.useState<number | null>(null);
  const [addedMessage, setAddedMessage] = React.useState<string | null>(null);

  const item = useQuery({
    queryKey: ["menu-item", itemId],
    queryFn: () => api.menuItem(itemId),
    enabled: Number.isFinite(itemId),
  });

  const similar = useQuery({
    queryKey: ["similar", itemId],
    queryFn: () => api.similar(itemId, 8),
    enabled: Number.isFinite(itemId),
  });

  // Pre-fill the widget if this dish is already rated, so the page shows the
  // customer's own opinion rather than an empty row of stars.
  const myRatings = useQuery({
    queryKey: ["my-ratings"],
    queryFn: () => api.myRatings(100),
    enabled: Boolean(user),
  });

  React.useEffect(() => {
    const existing = myRatings.data?.items.find(
      (entry) => entry.food_item_id === itemId,
    );
    if (existing) setMyRating(Math.round(Number.parseFloat(existing.rating)));
  }, [myRatings.data, itemId]);

  const rate = useMutation({
    mutationFn: (value: number) => api.rate(itemId, value),
    onSuccess: (_data, value) => {
      setMyRating(value);
      // The feed is derived from ratings, so it has to be refetched or the
      // customer sees an unchanged list and assumes the rating was ignored.
      void queryClient.invalidateQueries({ queryKey: ["recommendations"] });
      void queryClient.invalidateQueries({ queryKey: ["menu-item", itemId] });
      void queryClient.invalidateQueries({ queryKey: ["my-ratings"] });
    },
  });

  if (item.isLoading) {
    return (
      <div className="grid gap-8 md:grid-cols-2">
        <Skeleton className="aspect-[4/3] w-full" />
        <div className="space-y-4">
          <Skeleton className="h-8 w-2/3" />
          <Skeleton className="h-4 w-full" />
          <Skeleton className="h-4 w-5/6" />
          <Skeleton className="h-12 w-40" />
        </div>
      </div>
    );
  }

  if (item.isError || !item.data) {
    const notFound = item.error instanceof ApiError && item.error.status === 404;
    return (
      <div className="space-y-4">
        <Alert>
          {notFound
            ? "That dish is not on the menu."
            : item.error instanceof Error
              ? item.error.message
              : "Could not load this dish."}
        </Alert>
        <Link href="/menu">
          <Button variant="secondary">
            <ArrowLeft aria-hidden="true" /> Back to the menu
          </Button>
        </Link>
      </div>
    );
  }

  const dish = item.data;
  const style = cuisineStyle(dish.cuisine);

  return (
    <div className="space-y-12">
      <Link
        href="/menu"
        className="inline-flex items-center gap-1 text-sm text-muted-foreground hover:text-foreground"
      >
        <ArrowLeft className="size-4" aria-hidden="true" /> Back to the menu
      </Link>

      <div className="grid gap-8 md:grid-cols-2">
        <DishImage
          name={dish.name}
          cuisine={dish.cuisine}
          imageUrl={dish.image_url}
          className="aspect-[4/3] w-full rounded-lg"
          glyphClassName="text-8xl"
        />

        <div className="space-y-5">
          <div className="space-y-2">
            <span
              className={`inline-block rounded-full px-2.5 py-1 text-xs font-medium ${style.badge}`}
            >
              {humanise(dish.cuisine)}
            </span>
            <h1 className="text-3xl font-bold tracking-tight">{dish.name}</h1>
            <div className="flex flex-wrap items-center gap-3">
              <RatingStars value={dish.average_rating} count={dish.rating_count} />
              {dish.restaurant ? (
                <span className="inline-flex items-center gap-1 text-xs text-muted-foreground">
                  <MapPin className="size-3.5" aria-hidden="true" />
                  {dish.restaurant.name}, {dish.restaurant.area}
                </span>
              ) : null}
            </div>
          </div>

          <p className="text-muted-foreground">{dish.description}</p>

          <div className="flex flex-wrap gap-2">
            {dish.is_veg ? (
              <Badge className="bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300">
                <Leaf className="size-3" aria-hidden="true" /> Vegetarian
              </Badge>
            ) : null}
            <Badge>
              <Flame className="size-3" aria-hidden="true" />
              {SPICE_LABELS[dish.spice_level]}
            </Badge>
            <Badge>
              <Clock className="size-3" aria-hidden="true" />
              {dish.prep_time_min} min
            </Badge>
            {dish.is_rice_based ? <Badge>Rice based</Badge> : null}
          </div>

          {dish.ingredient_tags.length ? (
            <div>
              <h2 className="text-xs font-medium uppercase tracking-wide text-muted-foreground">
                Made with
              </h2>
              <p className="mt-1 text-sm">
                {dish.ingredient_tags.map(humanise).join(" · ")}
              </p>
            </div>
          ) : null}

          <div className="flex items-end justify-between gap-4 border-t border-border pt-5">
            <div>
              <p className="text-xs uppercase tracking-wide text-muted-foreground">
                Price
              </p>
              <p className="text-3xl font-bold text-accent">
                {formatPrice(dish.price)}
              </p>
            </div>

            <div className="flex items-center gap-3">
              <div className="flex items-center rounded-md border border-border">
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setQuantity((value) => Math.max(1, value - 1))}
                  aria-label="Decrease quantity"
                  disabled={quantity <= 1}
                >
                  <Minus />
                </Button>
                <span className="w-10 text-center text-sm font-medium" aria-live="polite">
                  {quantity}
                </span>
                <Button
                  variant="ghost"
                  size="icon"
                  onClick={() => setQuantity((value) => Math.min(20, value + 1))}
                  aria-label="Increase quantity"
                >
                  <Plus />
                </Button>
              </div>

              <Button
                size="lg"
                disabled={!dish.is_available}
                onClick={() => {
                  add(dish, quantity);
                  setAddedMessage(
                    `${quantity} × ${dish.name} added to your cart.`,
                  );
                  window.setTimeout(() => setAddedMessage(null), 2500);
                }}
              >
                {dish.is_available ? "Add to cart" : "Unavailable"}
              </Button>
            </div>
          </div>

          <p aria-live="polite" className="min-h-5 text-sm text-success">
            {addedMessage}
          </p>
        </div>
      </div>

      <Card className="space-y-3 p-5">
        <h2 className="text-lg font-semibold">Rate this dish</h2>
        {user ? (
          <>
            <p className="text-sm text-muted-foreground">
              Your rating feeds straight back into what gets recommended to you.
            </p>
            <RatingInput
              value={myRating}
              onChange={(value) => rate.mutate(value)}
              disabled={rate.isPending}
            />
            {rate.isSuccess ? (
              <p className="text-sm text-success" role="status">
                Saved — your recommendations have been updated.
              </p>
            ) : null}
            {rate.isError ? (
              <Alert>
                {rate.error instanceof Error
                  ? rate.error.message
                  : "Could not save your rating."}
              </Alert>
            ) : null}
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            <Link href="/login" className="font-medium text-accent hover:underline">
              Sign in
            </Link>{" "}
            to rate this dish and get recommendations built from your own taste.
          </p>
        )}
      </Card>

      <DishRow
        title="Similar dishes you may like"
        subtitle="Nearest neighbours in the model's learned feature space — dishes people rate the way they rate this one."
        items={similar.data?.items ?? []}
        isLoading={similar.isLoading}
        emptyMessage="No similar dishes indexed for this one yet."
      />
    </div>
  );
}
