'use client'

import * as React from 'react'
import * as ProgressPrimitive from '@radix-ui/react-progress'

import { cn } from '@/lib/utils'

type ProgressProps = React.ComponentProps<typeof ProgressPrimitive.Root> & {
  value?: number
  variant?: 'default' | 'temperature'
  animated?: boolean
  highlight?: boolean
}

function Progress({
  className,
  value = 0,
  variant = 'default',
  animated = true,
  highlight = false,
  ...props
}: ProgressProps) {
  const clamped = Math.max(0, Math.min(100, value))
  const t = clamped / 100
  const hue = Math.round(210 * (1 - t))
  const highlightColor = `hsl(${hue}, 90%, 55%)`

  return (
    <ProgressPrimitive.Root
      data-slot="progress"
      className={cn(
        variant === 'temperature'
          ? 'bg-gradient-to-r from-blue-500/20 via-amber-400/20 to-red-500/20'
          : 'bg-primary/20',
        'relative h-2 w-full overflow-hidden rounded-full',
        className,
      )}
      {...props}
    >
      <ProgressPrimitive.Indicator
        data-slot="progress-indicator"
        className={cn(
          variant === 'temperature'
            ? 'bg-gradient-to-r from-blue-500 via-amber-400 to-red-500'
            : 'bg-primary',
          animated
            ? 'transition-[transform,box-shadow,opacity] duration-700 ease-[cubic-bezier(0.22,1,0.36,1)]'
            : 'transition-none',
          'h-full w-full flex-1 relative',
        )}
        style={{ transform: `translateX(-${100 - clamped}%)` }}
      >
        {highlight && variant === 'temperature' ? (
          <div
            aria-hidden
            className="absolute right-0 top-1/2 -translate-y-1/2 translate-x-1/2 rounded-full"
            style={{
              width: 12,
              height: 12,
              backgroundColor: highlightColor,
              boxShadow: `0 0 16px ${highlightColor}, 0 0 4px ${highlightColor}`,
            }}
          />
        ) : null}
      </ProgressPrimitive.Indicator>
    </ProgressPrimitive.Root>
  )
}

export { Progress }
