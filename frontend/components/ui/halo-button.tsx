"use client"

import * as React from "react"
import { cn } from "@/lib/utils"

interface HaloButtonProps extends React.ButtonHTMLAttributes<HTMLButtonElement> {
  isActive?: boolean
  icon?: React.ReactNode
  children?: React.ReactNode
}

export const HaloButton = React.forwardRef<HTMLButtonElement, HaloButtonProps>(
  ({ className, isActive, icon, children, ...props }, ref) => {
    const handleMouseMove = (e: React.MouseEvent<HTMLButtonElement>) => {
      const rect = e.currentTarget.getBoundingClientRect()
      const x = e.clientX - rect.left
      const y = e.clientY - rect.top
      e.currentTarget.style.setProperty("--x", `${x}px`)
      e.currentTarget.style.setProperty("--y", `${y}px`)
    }

    return (
      <button
        ref={ref}
        onMouseMove={handleMouseMove}
        className={cn(
          "group relative w-full overflow-hidden rounded-xl border border-zinc-100 bg-white transition-all hover:border-zinc-200 hover:shadow-sm active:scale-[0.98]",
          isActive ? "border-primary/20 ring-1 ring-primary/10" : "",
          className
        )}
        {...props}
      >
        {/* Gradient Vignette / Background */}
        <div className="absolute inset-0 bg-gradient-to-tr from-zinc-50/50 via-white to-white opacity-100" />

        {/* Sliding Halo Overlay */}
        <div
          className="pointer-events-none absolute -inset-px opacity-0 transition-opacity duration-500 group-hover:opacity-100"
          style={{
            background: `radial-gradient(600px circle at var(--x, 0px) var(--y, 0px), rgba(0,0,0,0.04), transparent 40%)`,
          }}
        />

        {/* Thin Top Line */}
        <div className="absolute left-0 top-0 h-px w-full bg-gradient-to-r from-transparent via-zinc-300/60 to-transparent opacity-0 transition-opacity duration-300 group-hover:opacity-100" />

        {/* Content */}
        <div className="relative z-10 flex items-center gap-3 px-4 py-3">
          {icon && (
            <div className="flex h-8 w-8 items-center justify-center rounded-lg border border-zinc-100 bg-white text-zinc-500 shadow-sm transition-colors group-hover:border-zinc-200 group-hover:text-zinc-700">
              {icon}
            </div>
          )}
          <span className="text-sm font-medium text-zinc-600 transition-colors group-hover:text-zinc-900">
            {children}
          </span>
        </div>
      </button>
    )
  }
)
HaloButton.displayName = "HaloButton"
