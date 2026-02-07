'use client'

import useSWR from 'swr'
import { useCallback, useEffect, useState } from 'react'
import { getFingerprint } from '@/lib/fingerprint'

interface VoteData {
  votes: { true: number; false: number }
  userVote: { vote: 'true' | 'false'; reasoning?: string } | null
}

const fetcher = (url: string) => fetch(url).then(res => res.json())

export function useVote(day: number, phase: string) {
  const [fingerprint, setFingerprint] = useState<string>('')
  // Local optimistic state for immediate UI feedback
  const [localVote, setLocalVote] = useState<'true' | 'false' | null>(null)
  const [localVotes, setLocalVotes] = useState<{ true: number; false: number } | null>(null)
  
  useEffect(() => {
    const fp = getFingerprint()
    setFingerprint(fp)
  }, [])
  
  const url = `/api/votes?day=${day}&phase=${phase}&fingerprint=${fingerprint || 'anon'}`
  
  const { data, error, isLoading, mutate } = useSWR<VoteData>(
    url,
    fetcher,
    { revalidateOnFocus: false, dedupingInterval: 30000 }
  )
  
  // Sync server data to local state when it arrives
  useEffect(() => {
    if (data?.userVote?.vote && !localVote) {
      setLocalVote(data.userVote.vote)
    }
    if (data?.votes && !localVotes) {
      setLocalVotes(data.votes)
    }
  }, [data, localVote, localVotes])
  
  const submitVote = useCallback(async (vote: 'true' | 'false', reasoning?: string) => {
    console.log('[v0] submitVote called with:', vote, 'day:', day, 'phase:', phase)
    
    // IMMEDIATELY update local state for instant feedback
    const previousVote = localVote
    const previousVotes = localVotes || data?.votes || { true: 70, false: 30 }
    console.log('[v0] Previous state:', { previousVote, previousVotes })
    
    // Calculate new vote counts
    let newVotes = { ...previousVotes }
    if (previousVote && previousVote !== vote) {
      // Changing vote
      if (previousVote === 'true') { newVotes.true--; newVotes.false++ }
      else { newVotes.false--; newVotes.true++ }
    } else if (!previousVote) {
      // New vote
      if (vote === 'true') newVotes.true++
      else newVotes.false++
    }
    
    // Optimistic update - instant UI feedback
    console.log('[v0] Setting optimistic state:', { vote, newVotes })
    setLocalVote(vote)
    setLocalVotes(newVotes)
    
    // Then persist to server in background
    const fp = fingerprint || getFingerprint()
    try {
      const response = await fetch('/api/votes', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ day, phase, fingerprint: fp, vote, reasoning })
      })
      
      if (response.ok) {
        const result = await response.json()
        // Update with server-confirmed data
        setLocalVotes(result.votes)
        mutate({ votes: result.votes, userVote: result.userVote }, false)
      }
    } catch (err) {
      // Keep optimistic state even on error - better UX
      console.error('[v0] Vote submission error:', err)
    }
    
    return { vote, votes: newVotes }
  }, [day, phase, fingerprint, localVote, localVotes, data?.votes, mutate])
  
  // Use local state if available, fall back to server data
  const votes = localVotes || data?.votes || { true: 70, false: 30 }
  const userVote = localVote || data?.userVote?.vote || null
  
  const trueCount = votes.true
  const falseCount = votes.false
  const totalVotes = trueCount + falseCount
  const truePercent = totalVotes > 0 ? Math.round((trueCount / totalVotes) * 100) : 70
  
  return {
    votes,
    userVote,
    userReasoning: data?.userVote?.reasoning || null,
    truePercent,
    totalVotes,
    isLoading,
    error,
    submitVote,
    refresh: mutate
  }
}
