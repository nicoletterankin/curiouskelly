'use client'

import useSWR from 'swr'
import { useCallback, useEffect, useState, useRef } from 'react'
import { getFingerprint, generateUsername } from '@/lib/fingerprint'

interface Comment {
  id: string
  username: string
  message: string
  vote?: 'true' | 'false' | null
  language: string
  created_at: string
  phase?: string
  pending?: boolean // Local optimistic comment
}

interface CommentsData {
  comments: Comment[]
  votes: { true: number; false: number }
  total: number
  seed?: boolean
  cached?: boolean
}

// Seed comments for offline/error states
const SEED_COMMENTS: Comment[] = [
  { id: 'seed_1', username: 'curious_learner', message: 'Love learning with Kelly!', language: 'en', created_at: new Date().toISOString() },
  { id: 'seed_2', username: 'science_mom', message: 'My kids are obsessed with these lessons', language: 'en', created_at: new Date().toISOString() },
  { id: 'seed_3', username: 'wonder_seeker', message: 'Mind = blown', language: 'en', created_at: new Date().toISOString() },
]

const fetcher = async (url: string): Promise<CommentsData> => {
  try {
    const controller = new AbortController()
    const timeout = setTimeout(() => controller.abort(), 5000)
    
    const res = await fetch(url, { signal: controller.signal })
    clearTimeout(timeout)
    
    if (!res.ok) {
      return { comments: SEED_COMMENTS, votes: { true: 0, false: 0 }, total: SEED_COMMENTS.length, seed: true }
    }
    return res.json()
  } catch {
    return { comments: SEED_COMMENTS, votes: { true: 0, false: 0 }, total: SEED_COMMENTS.length, seed: true }
  }
}

export function useComments(day: number, phase?: string) {
  const [fingerprint, setFingerprint] = useState<string>('')
  const [username, setUsername] = useState<string>('')
  const [localComments, setLocalComments] = useState<Comment[]>([])
  const pendingQueueRef = useRef<Comment[]>([])
  
  useEffect(() => {
    const fp = getFingerprint()
    setFingerprint(fp)
    setUsername(generateUsername(fp))
  }, [])
  
  const url = phase 
    ? `/api/comments?day=${day}&phase=${phase}&limit=30`
    : `/api/comments?day=${day}&limit=30`
  
  const { data, error, isLoading, mutate } = useSWR<CommentsData>(
    url,
    fetcher,
    { 
      refreshInterval: 30000, // Reduced to 30s to avoid rate limits
      revalidateOnFocus: false,
      errorRetryCount: 2,
      errorRetryInterval: 5000,
      fallbackData: { comments: SEED_COMMENTS, votes: { true: 0, false: 0 }, total: SEED_COMMENTS.length }
    }
  )
  
  // Process pending queue when online
  useEffect(() => {
    const processPending = async () => {
      if (pendingQueueRef.current.length === 0) return
      
      const comment = pendingQueueRef.current[0]
      try {
        const response = await fetch('/api/comments', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            day,
            phase: phase || 'general',
            fingerprint,
            username: comment.username,
            message: comment.message,
            language: 'en'
          })
        })
        
        if (response.ok) {
          pendingQueueRef.current.shift()
          setLocalComments(prev => prev.filter(c => c.id !== comment.id))
          mutate()
        }
      } catch {
        // Keep in queue
      }
    }
    
    const interval = setInterval(processPending, 15000)
    return () => clearInterval(interval)
  }, [day, phase, fingerprint, mutate])
  
  const postComment = useCallback(async (message: string, vote?: 'true' | 'false') => {
    if (!fingerprint || !message.trim()) return null
    
    // Create optimistic comment
    const optimisticComment: Comment = {
      id: `local_${Date.now()}`,
      username,
      message: message.trim(),
      vote,
      language: 'en',
      created_at: new Date().toISOString(),
      phase: phase || 'general',
      pending: true
    }
    
    // Add to local state immediately
    setLocalComments(prev => [optimisticComment, ...prev])
    
    try {
      const controller = new AbortController()
      const timeout = setTimeout(() => controller.abort(), 5000)
      
      const response = await fetch('/api/comments', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          day,
          phase: phase || 'general',
          fingerprint,
          username,
          message: message.trim(),
          vote,
          language: 'en'
        }),
        signal: controller.signal
      })
      
      clearTimeout(timeout)
      
      if (!response.ok) throw new Error('Failed to post')
      
      const result = await response.json()
      
      // Replace optimistic with real
      setLocalComments(prev => prev.filter(c => c.id !== optimisticComment.id))
      mutate()
      
      return result.comment
    } catch {
      // Add to pending queue for retry
      pendingQueueRef.current.push(optimisticComment)
      // Keep optimistic comment visible
      return optimisticComment
    }
  }, [day, phase, fingerprint, username, mutate])
  
  // Merge local optimistic comments with server data
  const allComments = [...localComments, ...(data?.comments || [])]
  
  return {
    comments: allComments,
    votes: data?.votes || { true: 0, false: 0 },
    total: allComments.length,
    isLoading: isLoading && localComments.length === 0,
    error,
    postComment,
    fingerprint,
    username,
    refresh: mutate,
    isOffline: !!error,
    hasPending: localComments.length > 0
  }
}
