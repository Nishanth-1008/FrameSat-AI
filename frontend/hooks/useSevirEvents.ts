"use client";

import { useState } from "react";
import { useQuery } from "@tanstack/react-query";
import { fetchSevirEvents } from "@/services/api/sevir";

const PER_PAGE = 12;

export function useSevirEvents() {
  const [year, setYear] = useState<number | undefined>(undefined);
  const [page, setPage] = useState(1);

  const query = useQuery({
    queryKey: ["sevir-events", year, page],
    queryFn: () => fetchSevirEvents({ year, page, perPage: PER_PAGE }),
    staleTime: 60_000,
  });

  return {
    events: query.data?.events ?? [],
    total: query.data?.total ?? 0,
    page,
    perPage: PER_PAGE,
    setPage,
    year,
    setYear: (y: number | undefined) => {
      setYear(y);
      setPage(1);
    },
    isLoading: query.isLoading,
    isError: query.isError,
  };
}
