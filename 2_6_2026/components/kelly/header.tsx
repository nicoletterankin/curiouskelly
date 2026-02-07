'use client'

import { cn } from '@/lib/utils'
import { Button } from '@/components/ui/button'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuLabel,
  DropdownMenuSeparator,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'
import { Avatar, AvatarFallback, AvatarImage } from '@/components/ui/avatar'
import {
  Clock,
  Settings,
  LogOut,
  User,
  CreditCard,
  Menu,
  Crown,
} from 'lucide-react'

interface HeaderProps {
  // User data
  displayName: string
  email: string | null
  isAuthenticated: boolean
  subscriptionStatus?: string | null
  
  // Date/time (ALWAYS full format)
  formattedDate: string      // "Friday, January 17, 2026"
  formattedTime: string      // "3:45 PM PST"
  greeting: string           // "Good morning"
  
  // Actions
  onMenuClick?: () => void
  onSignIn?: () => void
  onSignOut?: () => void
  
  className?: string
}

export function Header({
  displayName,
  email,
  isAuthenticated,
  subscriptionStatus,
  formattedDate,
  formattedTime,
  greeting,
  onMenuClick,
  onSignIn,
  onSignOut,
  className,
}: HeaderProps) {
  const initials = displayName
    .split(' ')
    .map((n) => n[0])
    .join('')
    .toUpperCase()
    .slice(0, 2)

  const isPremium = subscriptionStatus === 'active'

  return (
    <header
      className={cn(
        'flex items-center justify-between px-4 py-3 border-b border-border bg-background/80 backdrop-blur-sm',
        className
      )}
    >
      {/* Left: Logo + Menu Button (mobile) */}
      <div className="flex items-center gap-3">
        <Button
          variant="ghost"
          size="icon"
          onClick={onMenuClick}
          className="lg:hidden"
        >
          <Menu className="h-5 w-5" />
          <span className="sr-only">Open menu</span>
        </Button>

        {/* Logo */}
        <div className="flex items-center gap-2">
          <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center">
            <span className="text-primary-foreground font-bold text-sm">CK</span>
          </div>
          <span className="font-semibold text-lg hidden sm:inline">
            Curious Kelly
          </span>
        </div>
      </div>

      {/* Center: Date & Time - ALWAYS FULL FORMAT */}
      <div className="hidden md:flex flex-col items-center">
        <time className="text-sm font-medium text-foreground">
          {formattedDate}
        </time>
        <div className="flex items-center gap-1.5 text-xs text-muted-foreground">
          <Clock className="h-3 w-3" />
          <span>{formattedTime}</span>
        </div>
      </div>

      {/* Right: User Menu */}
      <div className="flex items-center gap-3">
        {/* Date/time on mobile (compact but still full) */}
        <div className="flex md:hidden flex-col items-end text-right">
          <time className="text-xs font-medium text-foreground truncate max-w-[140px]">
            {formattedDate}
          </time>
          <span className="text-xs text-muted-foreground">{formattedTime}</span>
        </div>

        {isAuthenticated ? (
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button variant="ghost" className="relative h-9 px-2 gap-2">
                <Avatar className="h-7 w-7">
                  <AvatarImage src="/placeholder.svg" alt={displayName} />
                  <AvatarFallback className="text-xs bg-primary text-primary-foreground">
                    {initials}
                  </AvatarFallback>
                </Avatar>
                <span className="hidden sm:inline text-sm font-medium max-w-[120px] truncate">
                  {displayName}
                </span>
                {isPremium && (
                  <Crown className="h-3.5 w-3.5 text-white/70" />
                )}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent align="end" className="w-56">
              <DropdownMenuLabel>
                <div className="flex flex-col">
                  <span className="font-medium">{greeting}, {displayName}</span>
                  {email && (
                    <span className="text-xs text-muted-foreground font-normal truncate">
                      {email}
                    </span>
                  )}
                </div>
              </DropdownMenuLabel>
              <DropdownMenuSeparator />
              <DropdownMenuItem>
                <User className="mr-2 h-4 w-4" />
                Profile
              </DropdownMenuItem>
              <DropdownMenuItem>
                <CreditCard className="mr-2 h-4 w-4" />
                Subscription
                {isPremium && (
                  <span className="ml-auto text-xs text-[var(--kelly-blue)]">Premium</span>
                )}
              </DropdownMenuItem>
              <DropdownMenuItem>
                <Settings className="mr-2 h-4 w-4" />
                Settings
              </DropdownMenuItem>
              <DropdownMenuSeparator />
              <DropdownMenuItem onClick={onSignOut} className="text-destructive">
                <LogOut className="mr-2 h-4 w-4" />
                Sign out
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        ) : (
          <Button onClick={onSignIn} size="sm" className="gap-2">
            <User className="h-4 w-4" />
            <span className="hidden sm:inline">Sign in</span>
          </Button>
        )}
      </div>
    </header>
  )
}

// Date/time display component for use elsewhere
interface DateTimeDisplayProps {
  formattedDate: string
  formattedTime: string
  timezoneAbbr?: string
  variant?: 'default' | 'compact' | 'full'
  className?: string
}

export function DateTimeDisplay({
  formattedDate,
  formattedTime,
  variant = 'default',
  className,
}: DateTimeDisplayProps) {
  if (variant === 'compact') {
    return (
      <div className={cn('flex items-center gap-2 text-sm', className)}>
        <time className="font-medium">{formattedDate}</time>
        <span className="text-muted-foreground">at</span>
        <span className="text-muted-foreground">{formattedTime}</span>
      </div>
    )
  }

  if (variant === 'full') {
    return (
      <div className={cn('text-center', className)}>
        <time className="block text-2xl font-semibold tracking-tight">
          {formattedDate}
        </time>
        <div className="flex items-center justify-center gap-1.5 mt-1 text-muted-foreground">
          <Clock className="h-4 w-4" />
          <span>{formattedTime}</span>
        </div>
      </div>
    )
  }

  // Default
  return (
    <div className={cn('flex flex-col', className)}>
      <time className="text-sm font-medium">{formattedDate}</time>
      <span className="text-xs text-muted-foreground">{formattedTime}</span>
    </div>
  )
}

// Lesson metadata header - shows lesson info with FULL date
interface LessonMetaHeaderProps {
  title: string
  formattedDate: string
  theme?: string
  quote?: string
  quoteAuthor?: string
  tags?: string[]
  className?: string
}

export function LessonMetaHeader({
  title,
  formattedDate,
  theme,
  quote,
  quoteAuthor,
  tags = [],
  className,
}: LessonMetaHeaderProps) {
  return (
    <div className={cn('space-y-3', className)}>
      {/* Attribution line */}
      <div className="flex items-center gap-2 text-sm text-muted-foreground">
        <span className="font-medium text-primary">@curiouskelly</span>
        <span>·</span>
        <time>{formattedDate}</time>
      </div>

      {/* Title */}
      <h1 className="text-2xl md:text-3xl font-bold tracking-tight text-balance">
        {title}
      </h1>

      {/* Theme/subtitle */}
      {theme && (
        <p className="text-muted-foreground text-lg">{theme}</p>
      )}

      {/* Tags */}
      {tags.length > 0 && (
        <div className="flex flex-wrap gap-2">
          {tags.map((tag) => (
            <span
              key={tag}
              className="inline-flex items-center px-2.5 py-0.5 rounded-full text-xs font-medium bg-secondary text-secondary-foreground"
            >
              #{tag}
            </span>
          ))}
        </div>
      )}

      {/* Quote */}
      {quote && (
        <blockquote className="border-l-2 border-primary pl-4 italic text-muted-foreground">
          &ldquo;{quote}&rdquo;
          {quoteAuthor && (
            <footer className="mt-1 text-sm not-italic">— {quoteAuthor}</footer>
          )}
        </blockquote>
      )}
    </div>
  )
}
