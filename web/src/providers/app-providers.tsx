"use client";

import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { useRouter } from "next/navigation";
import * as React from "react";

import { ApiError, api, getToken, setToken } from "@/lib/api";
import type { RegisterPayload, UserResponse } from "@/lib/types";

interface AuthContextValue {
  user: UserResponse | null;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  register: (payload: RegisterPayload) => Promise<void>;
  logout: () => void;
}

const AuthContext = React.createContext<AuthContextValue | null>(null);

export function useAuth(): AuthContextValue {
  const context = React.useContext(AuthContext);
  if (!context) throw new Error("useAuth must be used inside <AppProviders>");
  return context;
}

function AuthProvider({ children }: { children: React.ReactNode }) {
  const [user, setUser] = React.useState<UserResponse | null>(null);
  const [isLoading, setIsLoading] = React.useState(true);
  const router = useRouter();

  // Restore the session on mount. The token is only in localStorage, so this
  // cannot run during SSR and the first paint is deliberately logged-out.
  React.useEffect(() => {
    let cancelled = false;

    async function restore() {
      if (!getToken()) {
        setIsLoading(false);
        return;
      }
      try {
        const me = await api.me();
        if (!cancelled) setUser(me);
      } catch (error) {
        // An expired or tampered token should log the user out quietly rather
        // than leaving the app in a half-authenticated state.
        if (error instanceof ApiError && error.isUnauthorised) setToken(null);
      } finally {
        if (!cancelled) setIsLoading(false);
      }
    }

    void restore();
    return () => {
      cancelled = true;
    };
  }, []);

  const login = React.useCallback(async (email: string, password: string) => {
    const token = await api.login(email, password);
    setToken(token.access_token);
    setUser(await api.me());
  }, []);

  const register = React.useCallback(async (payload: RegisterPayload) => {
    const token = await api.register(payload);
    setToken(token.access_token);
    setUser(await api.me());
  }, []);

  const logout = React.useCallback(() => {
    setToken(null);
    setUser(null);
    router.push("/");
  }, [router]);

  const value = React.useMemo(
    () => ({ user, isLoading, login, register, logout }),
    [user, isLoading, login, register, logout],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function AppProviders({ children }: { children: React.ReactNode }) {
  const [client] = React.useState(
    () =>
      new QueryClient({
        defaultOptions: {
          queries: {
            staleTime: 60_000,
            // The menu does not change while someone is looking at it, and
            // refetching on every window focus makes the page flicker for no
            // benefit.
            refetchOnWindowFocus: false,
            retry: (failureCount, error) => {
              // Retrying a 401 or a 404 just delays the error the user needs
              // to see.
              if (error instanceof ApiError && error.status >= 400 && error.status < 500) {
                return false;
              }
              return failureCount < 2;
            },
          },
        },
      }),
  );

  return (
    <QueryClientProvider client={client}>
      <AuthProvider>{children}</AuthProvider>
    </QueryClientProvider>
  );
}
