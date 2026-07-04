"use client";

import { motion } from "framer-motion";
import { Satellite, MapPin, Calendar, ChevronLeft, ChevronRight } from "lucide-react";
import clsx from "clsx";
import { useSevirEvents } from "@/hooks/useSevirEvents";
import { useSevirStore } from "@/store/useSevirStore";

export function EventsBrowser() {
  const { events, total, page, perPage, setPage, year, setYear, isLoading, isError } =
    useSevirEvents();
  const { selectedEvent, selectEvent } = useSevirStore();

  const totalPages = Math.max(1, Math.ceil(total / perPage));

  return (
    <div className="flex h-full flex-col border-2 border-paper bg-ink p-4">
      <div className="mb-3 flex items-center justify-between border-b-2 border-paper pb-3">
        <h3 className="flex items-center gap-2 font-display text-sm font-black uppercase tracking-tight text-paper">
          <Satellite size={16} className="text-cyan" />
          Storm Events
        </h3>
        <span className="font-mono text-[10px] text-muted">{total} found</span>
      </div>

      <div className="mb-3 flex items-center gap-2">
        <Calendar size={14} className="text-muted" />
        <input
          type="number"
          placeholder="Filter by year (e.g. 2019)"
          value={year ?? ""}
          onChange={(e) =>
            setYear(e.target.value ? Number(e.target.value) : undefined)
          }
          className="w-full border-2 border-paper bg-panel px-2 py-1.5 font-mono text-xs text-paper placeholder:text-muted focus:border-cyan focus:outline-none"
        />
      </div>

      <div className="flex-1 space-y-2 overflow-y-auto pr-1">
        {isLoading && (
          <p className="font-mono text-xs text-muted">Loading events…</p>
        )}
        {isError && (
          <p className="font-mono text-xs text-alert">
            Could not reach the SEVIR dataset endpoint.
          </p>
        )}
        {!isLoading && !isError && events.length === 0 && (
          <p className="font-mono text-xs text-muted">No events match this filter.</p>
        )}

        {events.map((event) => {
          const active = selectedEvent?.eventId === event.eventId;
          return (
            <motion.button
              key={event.eventId}
              onClick={() => selectEvent(event)}
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              className={clsx(
                "w-full border-2 p-3 text-left transition-colors",
                active
                  ? "border-cyan bg-panel shadow-brutalist-cyan"
                  : "border-paper/40 bg-panel hover:border-paper",
              )}
            >
              <div className="flex items-center justify-between">
                <span className="font-mono text-xs font-bold text-paper">
                  {event.eventId}
                </span>
                <span className="font-mono text-[10px] text-muted">{event.date}</span>
              </div>
              {event.bbox && (
                <div className="mt-1 flex items-center gap-1 text-muted">
                  <MapPin size={10} />
                  <span className="font-mono text-[10px]">
                    {event.bbox.llcrnrlat.toFixed(1)}, {event.bbox.llcrnrlon.toFixed(1)}
                  </span>
                </div>
              )}
              <div className="mt-1 flex flex-wrap gap-1">
                {event.imgTypes.map((t) => (
                  <span
                    key={t}
                    className="border border-muted px-1 font-mono text-[9px] uppercase text-muted"
                  >
                    {t}
                  </span>
                ))}
              </div>
            </motion.button>
          );
        })}
      </div>

      <div className="mt-3 flex items-center justify-between border-t-2 border-paper pt-3">
        <button
          onClick={() => setPage(Math.max(1, page - 1))}
          disabled={page <= 1}
          className="flex items-center gap-1 border-2 border-paper px-2 py-1 font-mono text-[10px] text-paper disabled:opacity-30"
        >
          <ChevronLeft size={12} /> Prev
        </button>
        <span className="font-mono text-[10px] text-muted">
          Page {page} / {totalPages}
        </span>
        <button
          onClick={() => setPage(Math.min(totalPages, page + 1))}
          disabled={page >= totalPages}
          className="flex items-center gap-1 border-2 border-paper px-2 py-1 font-mono text-[10px] text-paper disabled:opacity-30"
        >
          Next <ChevronRight size={12} />
        </button>
      </div>
    </div>
  );
}
