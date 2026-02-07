'use client'

import { useState, useCallback } from 'react'
import useSWR from 'swr'

interface SearchResult {
  day: number
  topic: string
  title: string
  preview: string
  url: string
}

interface SearchData {
  results: SearchResult[]
  total: number
  query: string
}

const fetcher = (url: string) => fetch(url).then(res => res.json())

export function useSearch() {
  const [query, setQuery] = useState('')
  const [debouncedQuery, setDebouncedQuery] = useState('')
  
  // Debounce search
  const search = useCallback((q: string) => {
    setQuery(q)
    
    // Debounce for 300ms
    const timeout = setTimeout(() => {
      setDebouncedQuery(q)
    }, 300)
    
    return () => clearTimeout(timeout)
  }, [])
  
  const { data, error, isLoading } = useSWR<SearchData>(
    debouncedQuery.length >= 2 ? `/api/search?q=${encodeURIComponent(debouncedQuery)}` : null,
    fetcher
  )
  
  const clearSearch = useCallback(() => {
    setQuery('')
    setDebouncedQuery('')
  }, [])
  
  return {
    query,
    search,
    clearSearch,
    results: data?.results || [],
    total: data?.total || 0,
    isLoading,
    error,
    hasSearched: debouncedQuery.length >= 2
  }
}
