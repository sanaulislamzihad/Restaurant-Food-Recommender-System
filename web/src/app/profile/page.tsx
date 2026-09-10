"use client";

import { useQuery } from "@tanstack/react-query";
import { Info, Package, Star } from "lucide-react";
import Link from "next/link";
import { useRouter } from "next/navigation";
import * as React from "react";

import { CuisineChart } from "@/components/cuisine-chart";
import { DishImage } from "@/components/dish-image";
import { RatingStars } from "@/components/rating-stars";
import { Button } from "@/components/ui/button";
import { Badge, Card, EmptyState, Skeleton } from "@/components/ui/primitives";
import { api } from "@/lib/api";
import { formatDate, formatPrice, formatRelative, humanise } from "@/lib/utils";
import { useAuth } from "@/providers/app-providers";

type Tab = "orders" | "ratings";

export default function ProfilePage() {
  const { user, isLoading: authLoading } = useAuth();
  const router = useRouter();
  const [tab, setTab] = React.useState<Tab>("orders");

  React.useEffect(() => {
    if (!authLoading && !user) router.replace("/login?next=/profile");
  }, [authLoading, user, router]);

  const profile = useQuery({
    queryKey: ["taste-profile", user?.id],
    queryFn: () => api.tasteProfile(),
    enabled: Boolean(user),
  });

  const orders = useQuery({
    queryKey: ["orders", user?.id],
    queryFn: () => api.orderHistory(20),
    enabled: Boolean(user),
  });

  const ratings = useQuery({
    queryKey: ["my-ratings", user?.id],
    queryFn: () => api.myRatings(50),
    enabled: Boolean(user),
  });

  if (authLoading || !user) {
    return (
      <div className="space-y-4">
        <Skeleton className="h-10 w-56" />
        <Skeleton className="h-40 w-full" />
      </div>
    );
  }

  const taste = profile.data;

  return (
    <div className="space-y-8">
      <header className="flex flex-wrap items-center justify-between gap-4">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">{user.name}</h1>
          <p className="text-sm text-muted-foreground">
            {user.email}
            {user.area ? ` · ${user.area}` : ""} · member since{" "}
            {formatDate(user.created_at)}
          </p>
        </div>
        <Link href="/menu">
          <Button variant="secondary">Browse the menu</Button>
        </Link>
      </header>

      <section className="grid gap-4 sm:grid-cols-3">
        <Card className="p-4">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            Orders
          </p>
          <p className="mt-1 text-3xl font-bold tabular-nums">
            {taste?.total_orders ?? "—"}
          </p>
        </Card>
        <Card className="p-4">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            Dishes rated
          </p>
          <p className="mt-1 text-3xl font-bold tabular-nums">
            {taste?.total_ratings ?? "—"}
          </p>
        </Card>
        <Card className="p-4">
          <p className="text-xs uppercase tracking-wide text-muted-foreground">
            Average you give
          </p>
          <p className="mt-1 text-3xl font-bold tabular-nums">
            {taste?.average_rating_given != null
              ? `${taste.average_rating_given.toFixed(1)}★`
              : "—"}
          </p>
        </Card>
      </section>

      <Card className="space-y-4 p-5">
        <div>
          <h2 className="text-lg font-semibold">Your taste profile</h2>
          <p className="text-sm text-muted-foreground">
            What the recommender knows about you, and where it came from.
          </p>
        </div>

        {profile.isLoading ? (
          <Skeleton className="h-40 w-full" />
        ) : taste && taste.top_cuisines.length > 0 ? (
          <>
            <CuisineChart data={taste.top_cuisines} />
            <div className="flex flex-wrap gap-2 pt-1">
              <Badge>
                Spice tolerance: {user.spice_tolerance}/5
              </Badge>
              <Badge>
                Favourite: {humanise(taste.top_cuisines[0].cuisine)}
              </Badge>
            </div>
          </>
        ) : (
          <p className="text-sm text-muted-foreground">
            Nothing here yet — order a few dishes and your cuisine mix appears.
          </p>
        )}

        {taste?.is_cold_start ? (
          <div className="flex items-start gap-3 rounded-md bg-muted p-3 text-sm text-muted-foreground">
            <Info className="mt-0.5 size-4 shrink-0" aria-hidden="true" />
            <p>
              You have rated {taste.total_ratings}{" "}
              {taste.total_ratings === 1 ? "dish" : "dishes"}. Below five, the
              collaborative model has too little to work from, so your feed is
              still popularity-based rather than personal.
            </p>
          </div>
        ) : null}
      </Card>

      <section className="space-y-4">
        <div
          role="tablist"
          aria-label="Your activity"
          className="inline-flex rounded-md border border-border p-1"
        >
          {(
            [
              { id: "orders", label: "Order history", icon: Package },
              { id: "ratings", label: "Your ratings", icon: Star },
            ] as const
          ).map(({ id, label, icon: Icon }) => (
            <button
              key={id}
              role="tab"
              type="button"
              aria-selected={tab === id}
              onClick={() => setTab(id)}
              className={`inline-flex items-center gap-2 rounded px-4 py-2 text-sm font-medium transition-colors ${
                tab === id
                  ? "bg-accent text-accent-foreground"
                  : "text-muted-foreground hover:text-foreground"
              }`}
            >
              <Icon className="size-4" aria-hidden="true" />
              {label}
            </button>
          ))}
        </div>

        {tab === "orders" ? (
          orders.isLoading ? (
            <Skeleton className="h-40 w-full" />
          ) : (orders.data?.items.length ?? 0) === 0 ? (
            <EmptyState
              glyph="📦"
              title="No orders yet"
              description="Once you place an order it will show up here."
              action={
                <Link href="/menu">
                  <Button>Browse the menu</Button>
                </Link>
              }
            />
          ) : (
            <ul className="space-y-3">
              {orders.data!.items.map((order) => (
                <li key={order.id}>
                  <Card className="p-4">
                    <div className="flex flex-wrap items-center justify-between gap-2">
                      <div>
                        <p className="font-medium">Order #{order.id}</p>
                        <p className="text-xs text-muted-foreground">
                          {formatDate(order.created_at)} ·{" "}
                          {formatRelative(order.created_at)}
                        </p>
                      </div>
                      <div className="flex items-center gap-3">
                        <Badge
                          className={
                            order.status === "delivered"
                              ? "bg-emerald-100 text-emerald-800 dark:bg-emerald-950 dark:text-emerald-300"
                              : order.status === "cancelled"
                                ? "bg-red-100 text-red-800 dark:bg-red-950 dark:text-red-300"
                                : undefined
                          }
                        >
                          {humanise(order.status)}
                        </Badge>
                        <span className="font-semibold">
                          {formatPrice(order.total_amount)}
                        </span>
                      </div>
                    </div>
                    <p className="mt-2 text-sm text-muted-foreground">
                      {order.items.length}{" "}
                      {order.items.length === 1 ? "item" : "items"}
                    </p>
                  </Card>
                </li>
              ))}
            </ul>
          )
        ) : ratings.isLoading ? (
          <Skeleton className="h-40 w-full" />
        ) : (ratings.data?.items.length ?? 0) === 0 ? (
          <EmptyState
            glyph="⭐"
            title="You have not rated anything yet"
            description="Rating dishes is what turns the feed from popular into personal."
            action={
              <Link href="/menu">
                <Button>Find something to rate</Button>
              </Link>
            }
          />
        ) : (
          <ul className="grid gap-3 sm:grid-cols-2">
            {ratings.data!.items.map((entry) => (
              <li key={entry.id}>
                <Card className="flex items-center gap-3 p-3">
                  <Link href={`/menu/${entry.food_item.id}`} className="shrink-0">
                    <DishImage
                      name={entry.food_item.name}
                      cuisine={entry.food_item.cuisine}
                      imageUrl={entry.food_item.image_url}
                      className="size-14 rounded-md"
                      glyphClassName="text-2xl"
                    />
                  </Link>
                  <div className="min-w-0 flex-1">
                    <Link
                      href={`/menu/${entry.food_item.id}`}
                      className="line-clamp-1 text-sm font-medium hover:text-accent"
                    >
                      {entry.food_item.name}
                    </Link>
                    <div className="mt-0.5 flex items-center gap-2">
                      <RatingStars value={Number.parseFloat(entry.rating)} />
                      <span className="text-xs text-muted-foreground">
                        {formatRelative(entry.created_at)}
                      </span>
                    </div>
                  </div>
                </Card>
              </li>
            ))}
          </ul>
        )}
      </section>
    </div>
  );
}
