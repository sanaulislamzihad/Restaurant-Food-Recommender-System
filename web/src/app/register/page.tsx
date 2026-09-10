"use client";

import Link from "next/link";
import { useRouter } from "next/navigation";
import * as React from "react";

import { Button } from "@/components/ui/button";
import { Alert, Card, Input, Label, Select } from "@/components/ui/primitives";
import { SPICE_LABELS } from "@/lib/cuisine";
import type { Gender } from "@/lib/types";
import { useAuth } from "@/providers/app-providers";

const AREAS = [
  "Dhanmondi",
  "Gulshan",
  "Banani",
  "Uttara",
  "Mirpur",
  "Bashundhara",
  "Mohammadpur",
  "Old Dhaka",
];

const MIN_PASSWORD_LENGTH = 8;

export default function RegisterPage() {
  const { register } = useAuth();
  const router = useRouter();

  const [form, setForm] = React.useState({
    name: "",
    email: "",
    password: "",
    area: "",
    gender: "" as Gender | "",
    age: "",
    spice_tolerance: 2,
  });
  const [error, setError] = React.useState<string | null>(null);
  const [busy, setBusy] = React.useState(false);

  async function onSubmit(event: React.FormEvent) {
    event.preventDefault();
    setBusy(true);
    setError(null);
    try {
      await register({
        name: form.name,
        email: form.email,
        password: form.password,
        area: form.area || null,
        gender: form.gender || null,
        age: form.age ? Number(form.age) : null,
        spice_tolerance: form.spice_tolerance,
      });
      router.replace("/");
    } catch (caught) {
      setError(caught instanceof Error ? caught.message : "Could not sign up.");
    } finally {
      setBusy(false);
    }
  }

  return (
    <div className="mx-auto max-w-md py-8">
      <Card className="space-y-5 p-6">
        <div>
          <h1 className="text-2xl font-bold tracking-tight">Create an account</h1>
          <p className="mt-1 text-sm text-muted-foreground">
            A new account starts with no history, so the first recommendations
            you see are simply what is popular. Rate five dishes and they become
            yours.
          </p>
        </div>

        <form className="space-y-4" onSubmit={onSubmit}>
          <div>
            <Label htmlFor="name">Name</Label>
            <Input
              id="name"
              required
              autoComplete="name"
              className="mt-1"
              value={form.name}
              onChange={(event) => setForm({ ...form, name: event.target.value })}
            />
          </div>

          <div>
            <Label htmlFor="email">Email</Label>
            <Input
              id="email"
              type="email"
              required
              autoComplete="email"
              className="mt-1"
              value={form.email}
              onChange={(event) => setForm({ ...form, email: event.target.value })}
            />
          </div>

          <div>
            <Label htmlFor="password">Password</Label>
            <Input
              id="password"
              type="password"
              required
              minLength={MIN_PASSWORD_LENGTH}
              autoComplete="new-password"
              className="mt-1"
              value={form.password}
              onChange={(event) =>
                setForm({ ...form, password: event.target.value })
              }
            />
            <p className="mt-1 text-[11px] text-muted-foreground">
              At least {MIN_PASSWORD_LENGTH} characters.
            </p>
          </div>

          <div className="grid grid-cols-2 gap-3">
            <div>
              <Label htmlFor="area">Area</Label>
              <Select
                id="area"
                className="mt-1"
                value={form.area}
                onChange={(event) => setForm({ ...form, area: event.target.value })}
              >
                <option value="">Prefer not to say</option>
                {AREAS.map((area) => (
                  <option key={area} value={area}>
                    {area}
                  </option>
                ))}
              </Select>
            </div>

            <div>
              <Label htmlFor="age">Age</Label>
              <Input
                id="age"
                type="number"
                min={10}
                max={120}
                className="mt-1"
                value={form.age}
                onChange={(event) => setForm({ ...form, age: event.target.value })}
              />
            </div>
          </div>

          <div>
            <Label htmlFor="spice">
              Spice tolerance — {SPICE_LABELS[form.spice_tolerance]}
            </Label>
            <input
              id="spice"
              type="range"
              min={0}
              max={5}
              step={1}
              className="mt-2 w-full accent-[hsl(var(--accent))]"
              value={form.spice_tolerance}
              onChange={(event) =>
                setForm({ ...form, spice_tolerance: Number(event.target.value) })
              }
            />
            <p className="mt-1 text-[11px] text-muted-foreground">
              Used to keep dishes hotter than you like out of your feed.
            </p>
          </div>

          {error ? <Alert>{error}</Alert> : null}

          <Button type="submit" className="w-full" size="lg" disabled={busy}>
            {busy ? "Creating account…" : "Create account"}
          </Button>
        </form>

        <p className="text-center text-sm text-muted-foreground">
          Already have an account?{" "}
          <Link href="/login" className="font-medium text-accent hover:underline">
            Sign in
          </Link>
        </p>
      </Card>
    </div>
  );
}
