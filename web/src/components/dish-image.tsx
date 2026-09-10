"use client";

import * as React from "react";

import { dishGlyph, dishHue } from "@/lib/cuisine";
import { cn } from "@/lib/utils";

/**
 * A dish tile that always renders something deliberate.
 *
 * The seeded catalogue points `image_url` at files that do not ship with the
 * repo, so the fallback is the normal case rather than the error case: a
 * cuisine-tinted gradient with a glyph chosen from the dish name. If a real
 * photo is dropped into `public/images/food/` it is used instead, and a photo
 * that 404s falls back rather than leaving a torn-image icon.
 */
export function DishImage({
  name,
  cuisine,
  imageUrl,
  className,
  glyphClassName,
}: {
  name: string;
  cuisine: string;
  imageUrl?: string | null;
  className?: string;
  glyphClassName?: string;
}) {
  const [failed, setFailed] = React.useState(false);
  const showPhoto = Boolean(imageUrl) && !failed;

  return (
    <div
      className={cn(
        "dish-tile relative flex items-center justify-center overflow-hidden",
        className,
      )}
      // Only the hue is set here; `.dish-tile` picks lightness per theme.
      style={{ "--dish-h": dishHue(name, cuisine) } as React.CSSProperties}
    >
      {showPhoto ? (
        /* next/image would need every possible image host declared in
           next.config, and the API is free to return arbitrary URLs. */
        // eslint-disable-next-line @next/next/no-img-element
        <img
          src={imageUrl as string}
          alt={name}
          loading="lazy"
          onError={() => setFailed(true)}
          className="h-full w-full object-cover"
        />
      ) : (
        <>
          <span
            className={cn("select-none drop-shadow-sm", glyphClassName ?? "text-5xl")}
            aria-hidden="true"
          >
            {dishGlyph(name, cuisine)}
          </span>
          {/* A faint wash so light glyphs stay legible on light gradients. */}
          <span className="pointer-events-none absolute inset-0 bg-black/5 dark:bg-black/20" />
        </>
      )}
    </div>
  );
}
