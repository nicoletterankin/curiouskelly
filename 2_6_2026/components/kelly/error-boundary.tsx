'use client'

/**
 * KellyOS Error Boundary
 * Gracefully handles errors in Kelly learning components
 * Shows friendly "Kelly had a hiccup" message instead of crashing
 */

import React, { Component, ReactNode } from 'react'
import { AlertCircle, RefreshCw, Home } from 'lucide-react'
import { cn } from '@/lib/utils'

interface ErrorBoundaryProps {
  children: ReactNode
  fallback?: ReactNode
  onError?: (error: Error, errorInfo: React.ErrorInfo) => void
  className?: string
  variant?: 'full' | 'inline' | 'minimal'
}

interface ErrorBoundaryState {
  hasError: boolean
  error: Error | null
  errorInfo: React.ErrorInfo | null
}

export class KellyErrorBoundary extends Component<ErrorBoundaryProps, ErrorBoundaryState> {
  constructor(props: ErrorBoundaryProps) {
    super(props)
    this.state = {
      hasError: false,
      error: null,
      errorInfo: null
    }
  }

  static getDerivedStateFromError(error: Error): Partial<ErrorBoundaryState> {
    return { hasError: true, error }
  }

  componentDidCatch(error: Error, errorInfo: React.ErrorInfo) {
    this.setState({ errorInfo })
    
    // Log error for monitoring
    console.error('[KellyErrorBoundary] Error caught:', error)
    console.error('[KellyErrorBoundary] Component stack:', errorInfo.componentStack)
    
    // Call optional error handler
    this.props.onError?.(error, errorInfo)
  }

  handleRetry = () => {
    this.setState({ hasError: false, error: null, errorInfo: null })
  }

  handleGoHome = () => {
    window.location.href = '/'
  }

  render() {
    if (this.state.hasError) {
      // Use custom fallback if provided
      if (this.props.fallback) {
        return this.props.fallback
      }

      const { variant = 'full', className } = this.props

      // Minimal variant - just an icon with retry
      if (variant === 'minimal') {
        return (
          <div className={cn(
            "flex items-center justify-center p-4",
            className
          )}>
            <button
              onClick={this.handleRetry}
              className="flex items-center gap-2 px-3 py-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/70 text-sm transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              <span>Retry</span>
            </button>
          </div>
        )
      }

      // Inline variant - compact error message
      if (variant === 'inline') {
        return (
          <div className={cn(
            "flex items-center gap-3 p-4 rounded-xl bg-amber-500/10 border border-amber-500/20",
            className
          )}>
            <AlertCircle className="w-5 h-5 text-amber-400 flex-shrink-0" />
            <div className="flex-1 min-w-0">
              <p className="text-sm text-white/90">Something went wrong</p>
              <p className="text-xs text-white/50 truncate">
                {this.state.error?.message || 'Unknown error'}
              </p>
            </div>
            <button
              onClick={this.handleRetry}
              className="p-2 rounded-lg bg-white/5 hover:bg-white/10 text-white/70 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
            </button>
          </div>
        )
      }

      // Full variant - centered error display
      return (
        <div className={cn(
          "flex flex-col items-center justify-center min-h-[300px] p-8 text-center",
          className
        )}>
          {/* Kelly avatar with error state */}
          <div className="relative mb-6">
            <div className="w-24 h-24 rounded-full bg-gradient-to-br from-amber-500/20 to-orange-500/20 border-2 border-amber-500/30 flex items-center justify-center">
              <AlertCircle className="w-10 h-10 text-amber-400" />
            </div>
            <div className="absolute -bottom-1 -right-1 w-8 h-8 rounded-full bg-background border-2 border-amber-500/30 flex items-center justify-center">
              <span className="text-lg">?</span>
            </div>
          </div>

          <h3 className="text-xl font-semibold text-white mb-2">
            Kelly had a hiccup!
          </h3>
          <p className="text-white/60 text-sm max-w-sm mb-6">
            Something unexpected happened while loading this part of your lesson. 
            Don't worry, your progress is saved!
          </p>

          {/* Error details (collapsed) */}
          {process.env.NODE_ENV === 'development' && this.state.error && (
            <details className="mb-6 w-full max-w-md text-left">
              <summary className="text-xs text-white/40 cursor-pointer hover:text-white/60">
                Technical details (dev only)
              </summary>
              <pre className="mt-2 p-3 rounded-lg bg-black/30 text-xs text-red-300 overflow-auto max-h-32">
                {this.state.error.message}
                {this.state.errorInfo?.componentStack}
              </pre>
            </details>
          )}

          {/* Action buttons */}
          <div className="flex items-center gap-3">
            <button
              onClick={this.handleRetry}
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-primary text-primary-foreground font-medium text-sm hover:bg-primary/90 transition-colors"
            >
              <RefreshCw className="w-4 h-4" />
              Try Again
            </button>
            <button
              onClick={this.handleGoHome}
              className="flex items-center gap-2 px-4 py-2 rounded-full bg-white/5 text-white/70 text-sm hover:bg-white/10 transition-colors"
            >
              <Home className="w-4 h-4" />
              Go Home
            </button>
          </div>
        </div>
      )
    }

    return this.props.children
  }
}

/**
 * Hook-based error boundary wrapper for functional components
 * Usage: <ErrorBoundaryWrapper><YourComponent /></ErrorBoundaryWrapper>
 */
export function ErrorBoundaryWrapper({
  children,
  onError,
  variant = 'full',
  className
}: ErrorBoundaryProps) {
  return (
    <KellyErrorBoundary
      onError={onError}
      variant={variant}
      className={className}
    >
      {children}
    </KellyErrorBoundary>
  )
}

/**
 * Phase-specific error boundary with context
 */
export function PhaseErrorBoundary({
  children,
  phase,
  dayOfYear
}: {
  children: ReactNode
  phase: string
  dayOfYear: number
}) {
  const handleError = (error: Error) => {
    // Log phase-specific error context
    console.error(`[PhaseError] Day ${dayOfYear}, Phase: ${phase}`, error)
  }

  return (
    <KellyErrorBoundary
      onError={handleError}
      variant="inline"
      fallback={
        <div className="flex flex-col items-center justify-center p-6 text-center">
          <AlertCircle className="w-8 h-8 text-amber-400 mb-3" />
          <p className="text-sm text-white/70">
            Couldn't load the {phase} phase
          </p>
          <button
            onClick={() => window.location.reload()}
            className="mt-3 text-xs text-primary hover:underline"
          >
            Refresh page
          </button>
        </div>
      }
    >
      {children}
    </KellyErrorBoundary>
  )
}

export default KellyErrorBoundary
