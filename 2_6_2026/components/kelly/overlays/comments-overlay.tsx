'use client'

import React from "react"
import { useState, useCallback } from 'react'
import { cn } from '@/lib/utils'
import { OverlayBackdrop, OverlayPanel, AnimatedList, AnimatedItem } from './index'
import { Heart, MessageCircle, Send, MoreHorizontal, Flag, Trash2 } from 'lucide-react'
import { Button } from '@/components/ui/button'
import { Input } from '@/components/ui/input'
import {
  DropdownMenu,
  DropdownMenuContent,
  DropdownMenuItem,
  DropdownMenuTrigger,
} from '@/components/ui/dropdown-menu'

interface Comment {
  id: string
  username: string
  avatar?: string
  text: string
  timestamp: string
  likes: number
  isLiked?: boolean
  isPinned?: boolean
  replies?: Comment[]
}

interface CommentsOverlayProps {
  isOpen: boolean
  onClose: () => void
  lessonTitle: string
  comments: Comment[]
  onAddComment?: (text: string) => void
  onLikeComment?: (commentId: string) => void
}

// Mock comments for demo
const MOCK_COMMENTS: Comment[] = [
  {
    id: '1',
    username: 'curious_mind_3',
    text: 'The way Kelly opened with "Built" completely grabbed my attention! Such a great hook.',
    timestamp: '2m ago',
    likes: 24,
    isPinned: true,
    isLiked: true,
  },
  {
    id: '2',
    username: 'daily_learner',
    text: 'I never thought about it from this angle before. This is why I tune in every day.',
    timestamp: '5m ago',
    likes: 18,
  },
  {
    id: '3',
    username: 'knowledge_seeker',
    text: 'This is exactly why I subscribe - unexpected perspectives that make you think differently.',
    timestamp: '8m ago',
    likes: 12,
    replies: [
      {
        id: '3-1',
        username: 'curious_mind_3',
        text: 'Totally agree! Kelly has a way of making everything fascinating.',
        timestamp: '6m ago',
        likes: 5,
      },
    ],
  },
  {
    id: '4',
    username: 'grateful_learner',
    text: 'Another day, another perspective shift. Thank you Kelly!',
    timestamp: '12m ago',
    likes: 8,
  },
  {
    id: '5',
    username: 'science_fan_22',
    text: 'The way she explains complex concepts in simple terms is amazing. My kids love watching with me.',
    timestamp: '15m ago',
    likes: 31,
  },
]

function CommentItem({
  comment,
  isLiked,
  onLike,
  onReplyLike,
  likedComments,
}: {
  comment: Comment
  isLiked: boolean
  onLike: () => void
  onReplyLike: (replyId: string) => void
  likedComments: Set<string>
}) {
  const [showReplies, setShowReplies] = useState(false)

  return (
    <div className={cn(
      'space-y-3',
      comment.isPinned && 'bg-primary/5 -mx-4 px-4 py-3 rounded-none border-l-2 border-primary'
    )}>
      {comment.isPinned && (
        <div className="text-xs text-primary font-medium">Pinned comment</div>
      )}
      
      <div className="flex gap-3">
        {/* Avatar */}
        <div className="w-8 h-8 rounded-full bg-muted flex items-center justify-center text-muted-foreground text-sm font-bold flex-shrink-0">
          {comment.avatar ? (
            <img src={comment.avatar || "/placeholder.svg"} alt="" className="w-full h-full rounded-full object-cover" />
          ) : (
            comment.username[0].toUpperCase()
          )}
        </div>

        <div className="flex-1 min-w-0">
          {/* Header */}
          <div className="flex items-center gap-2">
            <span className="font-semibold text-sm">{comment.username}</span>
            <span className="text-xs text-muted-foreground">{comment.timestamp}</span>
          </div>

          {/* Text */}
          <p className="text-sm text-foreground mt-1">{comment.text}</p>

          {/* Actions */}
          <div className="flex items-center gap-4 mt-2">
            <button
              onClick={onLike}
              className={cn(
                'flex items-center gap-1 text-xs transition-colors',
                isLiked ? 'text-white' : 'text-white/50 hover:text-white'
              )}
            >
              <Heart className={cn('h-3.5 w-3.5', isLiked && 'fill-white')} />
              <span>{comment.likes + (isLiked && !comment.isLiked ? 1 : 0)}</span>
            </button>
            
            <button
              onClick={() => setShowReplies(!showReplies)}
              className="flex items-center gap-1 text-xs text-muted-foreground hover:text-foreground transition-colors"
            >
              <MessageCircle className="h-3.5 w-3.5" />
              <span>Reply{comment.replies?.length ? ` (${comment.replies.length})` : ''}</span>
            </button>

            <DropdownMenu>
              <DropdownMenuTrigger asChild>
                <button className="text-muted-foreground hover:text-foreground transition-colors">
                  <MoreHorizontal className="h-4 w-4" />
                </button>
              </DropdownMenuTrigger>
              <DropdownMenuContent align="start">
                <DropdownMenuItem>
                  <Flag className="h-4 w-4 mr-2" />
                  Report
                </DropdownMenuItem>
                {comment.username === 'you' && (
                  <DropdownMenuItem className="text-destructive">
                    <Trash2 className="h-4 w-4 mr-2" />
                    Delete
                  </DropdownMenuItem>
                )}
              </DropdownMenuContent>
            </DropdownMenu>
          </div>

          {/* Replies */}
          {showReplies && comment.replies && comment.replies.length > 0 && (
            <div className="mt-3 space-y-3 pl-4 border-l-2 border-border/50">
              {comment.replies.map((reply) => (
                <div key={reply.id} className="flex gap-2">
                  <div className="w-6 h-6 rounded-full bg-muted flex items-center justify-center text-muted-foreground text-xs font-bold flex-shrink-0">
                    {reply.username[0].toUpperCase()}
                  </div>
                  <div className="flex-1 min-w-0">
                    <div className="flex items-center gap-2">
                      <span className="font-semibold text-xs">{reply.username}</span>
                      <span className="text-xs text-muted-foreground">{reply.timestamp}</span>
                    </div>
                    <p className="text-xs text-foreground mt-0.5">{reply.text}</p>
                    <button
                      onClick={() => onReplyLike(reply.id)}
                      className={cn(
                        'flex items-center gap-1 text-xs mt-1 transition-colors',
                        likedComments.has(reply.id) ? 'text-white' : 'text-white/50 hover:text-white'
                      )}
                    >
                      <Heart className={cn('h-3 w-3', likedComments.has(reply.id) && 'fill-white')} />
                      <span>{reply.likes}</span>
                    </button>
                  </div>
                </div>
              ))}
            </div>
          )}
        </div>
      </div>
    </div>
  )
}

export function CommentsOverlay({
  isOpen,
  onClose,
  lessonTitle,
  comments = MOCK_COMMENTS,
  onAddComment,
  onLikeComment,
}: CommentsOverlayProps) {
  const [newComment, setNewComment] = useState('')
  const [likedComments, setLikedComments] = useState<Set<string>>(new Set(['1']))
  const [localComments, setLocalComments] = useState(comments)

  const handleLike = useCallback((commentId: string) => {
    setLikedComments(prev => {
      const next = new Set(prev)
      if (next.has(commentId)) {
        next.delete(commentId)
      } else {
        next.add(commentId)
      }
      return next
    })
    onLikeComment?.(commentId)
  }, [onLikeComment])

  const handleSubmit = useCallback((e: React.FormEvent) => {
    e.preventDefault()
    if (!newComment.trim()) return
    
    const comment: Comment = {
      id: Date.now().toString(),
      username: 'you',
      text: newComment,
      timestamp: 'Just now',
      likes: 0,
    }
    
    setLocalComments(prev => [comment, ...prev])
    setNewComment('')
    onAddComment?.(newComment)
  }, [newComment, onAddComment])

  if (!isOpen) return null

  const totalComments = localComments.length + localComments.reduce((acc, c) => acc + (c.replies?.length || 0), 0)

  return (
    <OverlayBackdrop isOpen={isOpen} onClose={onClose} side="right">
      <OverlayPanel
        title="Comments"
        subtitle={`${totalComments} comments on "${lessonTitle}"`}
        onClose={onClose}
      >
        <div className="flex flex-col h-full">
          {/* Comment input */}
          <form onSubmit={handleSubmit} className="p-4 border-b border-border/50">
            <div className="flex gap-2">
              <div className="w-8 h-8 rounded-full bg-primary flex items-center justify-center text-primary-foreground text-sm font-bold flex-shrink-0">
                Y
              </div>
              <div className="flex-1 relative">
                <Input
                  value={newComment}
                  onChange={(e) => setNewComment(e.target.value)}
                  placeholder="Add a comment..."
                  className="pr-10 bg-muted/30 border-border/50"
                />
                <Button
                  type="submit"
                  variant="ghost"
                  size="icon"
                  className="absolute right-1 top-1/2 -translate-y-1/2 h-7 w-7"
                  disabled={!newComment.trim()}
                >
                  <Send className="h-4 w-4" />
                </Button>
              </div>
            </div>
          </form>

          {/* Comments list */}
          <div className="flex-1 overflow-y-auto">
            <AnimatedList className="p-4 space-y-4" delay={0.05}>
              {localComments.map((comment) => (
                <AnimatedItem key={comment.id}>
                  <CommentItem
                    key={comment.id}
                    comment={comment}
                    isLiked={likedComments.has(comment.id)}
                    onLike={() => handleLike(comment.id)}
                    onReplyLike={(replyId) => handleLike(replyId)}
                    likedComments={likedComments}
                  />
                </AnimatedItem>
              ))}
            </AnimatedList>
          </div>
        </div>
      </OverlayPanel>
    </OverlayBackdrop>
  )
}
