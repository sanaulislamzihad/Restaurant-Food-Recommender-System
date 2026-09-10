"use client";

import type { CuisineAffinity } from "@/lib/types";
import { humanise } from "@/lib/utils";

/**
 * Orders per cuisine, as a horizontal bar chart.
 *
 * Built from plain elements rather than a charting library. At five bars a
 * library would add a few hundred kilobytes to render five divs, and hand-built
 * bars keep the real numbers in the DOM as text, which is better for screen
 * readers than an SVG with the values buried in path geometry.
 *
 * Design decisions worth stating:
 *
 * - **One hue, not one per cuisine.** This is a single measure compared across
 *   categories, so colour would be decoration carrying no information. Identity
 *   comes from the label beside each bar.
 * - The hue is `--chart-1`, a step validated against both the light and dark
 *   surfaces, rather than the UI accent, which is tuned for buttons and sits too
 *   light to read as a data mark on a dark background.
 * - Every bar is directly labelled, so the chart never has to be decoded against
 *   an axis, and there is no axis to be recessive about.
 */
export function CuisineChart({ data }: { data: CuisineAffinity[] }) {
  if (data.length === 0) return null;

  const max = Math.max(...data.map((entry) => entry.order_count), 1);

  return (
    <figure className="space-y-3">
      <figcaption className="sr-only">
        Number of orders you have placed in each cuisine
      </figcaption>

      <ul className="space-y-2.5">
        {data.map((entry) => {
          const percentage = (entry.order_count / max) * 100;
          return (
            <li key={entry.cuisine} className="space-y-1">
              <div className="flex items-baseline justify-between gap-3 text-sm">
                <span className="font-medium">{humanise(entry.cuisine)}</span>
                <span className="tabular-nums text-muted-foreground">
                  {entry.order_count}{" "}
                  {entry.order_count === 1 ? "order" : "orders"}
                  {entry.average_rating !== null ? (
                    <> · you rate it {entry.average_rating.toFixed(1)}★</>
                  ) : null}
                </span>
              </div>

              {/* The track is the surface; the fill is the only coloured mark.
                  Rounded on the data end only, anchored flat to the baseline. */}
              <div className="h-2.5 w-full overflow-hidden rounded-l-sm bg-muted">
                <div
                  className="h-full rounded-r-[4px] bg-chart"
                  style={{ width: `${Math.max(percentage, 2)}%` }}
                />
              </div>
            </li>
          );
        })}
      </ul>

      {/* The numbers are already on screen, so this is the table view rather
          than a duplicate of it - it just makes the relationship explicit for
          assistive technology. */}
      <table className="sr-only">
        <caption>Orders by cuisine</caption>
        <thead>
          <tr>
            <th scope="col">Cuisine</th>
            <th scope="col">Orders</th>
            <th scope="col">Your average rating</th>
          </tr>
        </thead>
        <tbody>
          {data.map((entry) => (
            <tr key={entry.cuisine}>
              <th scope="row">{humanise(entry.cuisine)}</th>
              <td>{entry.order_count}</td>
              <td>
                {entry.average_rating !== null
                  ? entry.average_rating.toFixed(1)
                  : "not rated"}
              </td>
            </tr>
          ))}
        </tbody>
      </table>
    </figure>
  );
}
