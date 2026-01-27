'use client';

import { useState, useEffect, useMemo, useDeferredValue } from 'react';
import { Learner } from '../../../lib/sis-types';
import { generateMockLearners, calculateStats } from '../../../lib/sis-mock-data';
import { filterLearners, getUniqueCountries } from '../../../lib/sis-utils';
import { StatsBar } from '../../../components/sis/StatsBar';
import { Toolbar } from '../../../components/sis/Toolbar';
import { LearnersTable } from '../../../components/sis/LearnersTable';
import { LearnerDetailPanel } from '../../../components/sis/LearnerDetailPanel';
import { ComposeMessageModal } from '../../../components/sis/ComposeMessageModal';
import { RefreshCw } from 'lucide-react';

export default function SISDashboard() {
  const [learners, setLearners] = useState<Learner[]>([]);
  const [selectedIds, setSelectedIds] = useState<Set<string>>(new Set());
  const [selectedLearner, setSelectedLearner] = useState<Learner | null>(null);
  const [isComposeOpen, setIsComposeOpen] = useState(false);
  const [searchQuery, setSearchQuery] = useState('');
  const [statusFilter, setStatusFilter] = useState('all');
  const [countryFilter, setCountryFilter] = useState('all');
  const [isLive, setIsLive] = useState(true);

  // Debounced search
  const deferredSearch = useDeferredValue(searchQuery);

  // Initialize mock data
  useEffect(() => {
    setLearners(generateMockLearners(50));
  }, []);

  // Real-time updates
  useEffect(() => {
    if (!isLive) return;
    
    const interval = setInterval(() => {
      setLearners(prev => prev.map(l => ({
        ...l,
        isOnline: l.lastSeenAt 
          ? (Date.now() - new Date(l.lastSeenAt).getTime()) < 5 * 60 * 1000
          : false,
      })));
    }, 5000);
    
    return () => clearInterval(interval);
  }, [isLive]);

  // Keyboard handler
  useEffect(() => {
    const handleKeyDown = (e: KeyboardEvent) => {
      if (e.key === 'Escape') {
        if (isComposeOpen) {
          setIsComposeOpen(false);
        } else if (selectedLearner) {
          setSelectedLearner(null);
        }
      }
    };
    
    window.addEventListener('keydown', handleKeyDown);
    return () => window.removeEventListener('keydown', handleKeyDown);
  }, [isComposeOpen, selectedLearner]);

  // Filtered learners
  const filteredLearners = useMemo(() => 
    filterLearners(learners, { search: deferredSearch, status: statusFilter, country: countryFilter }),
    [learners, deferredSearch, statusFilter, countryFilter]
  );

  // Stats
  const stats = useMemo(() => calculateStats(learners), [learners]);

  // Countries for filter
  const countries = useMemo(() => getUniqueCountries(learners), [learners]);

  // Selection handlers
  const handleSelect = (id: string) => {
    setSelectedIds(prev => {
      const next = new Set(prev);
      if (next.has(id)) {
        next.delete(id);
      } else {
        next.add(id);
      }
      return next;
    });
  };

  const handleSelectAll = () => {
    if (filteredLearners.every(l => selectedIds.has(l.id))) {
      setSelectedIds(new Set());
    } else {
      setSelectedIds(new Set(filteredLearners.map(l => l.id)));
    }
  };

  const handleClearSelection = () => setSelectedIds(new Set());

  // Message handlers
  const handleSendMessage = (subject: string, body: string, type: 'email' | 'push') => {
    console.log('Sending message:', { subject, body, type, recipients: Array.from(selectedIds) });
    // TODO: API call
    setIsComposeOpen(false);
    setSelectedIds(new Set());
  };

  // Export handler
  const handleExport = () => {
    const selected = learners.filter(l => selectedIds.has(l.id));
    const csv = [
      ['ID', 'Name', 'Email', 'Country', 'Status', 'Streak', 'Lessons'].join(','),
      ...selected.map(l => [
        l.id,
        l.name || '',
        l.email || '',
        l.country || '',
        l.subscriptionStatus,
        l.currentStreak,
        l.lessonsCompleted
      ].join(','))
    ].join('\n');
    
    const blob = new Blob([csv], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `learners-export-${new Date().toISOString().split('T')[0]}.csv`;
    a.click();
    URL.revokeObjectURL(url);
  };

  // Get selected learners for compose modal
  const selectedLearners = learners.filter(l => selectedIds.has(l.id));

  return (
    <div className="min-h-screen bg-gradient-to-br from-gray-950 via-gray-900 to-gray-950">
      <div className="max-w-[1600px] mx-auto p-6 space-y-6">
        {/* Header */}
        <div className="flex items-center justify-between">
          <div>
            <h1 className="text-2xl font-bold text-white flex items-center gap-2">
              Curious Kelly
            </h1>
            <p className="text-gray-400">Student Information System</p>
          </div>
          <div className="flex items-center gap-3">
            <button
              onClick={() => setIsLive(!isLive)}
              className={`flex items-center gap-2 px-3 py-1.5 rounded-lg text-sm font-medium transition-colors ${
                isLive ? 'bg-green-500/20 text-green-400' : 'bg-gray-700 text-gray-400'
              }`}
            >
              <span className={`w-2 h-2 rounded-full ${isLive ? 'bg-green-400 animate-pulse' : 'bg-gray-500'}`}></span>
              {isLive ? 'Live' : 'Paused'}
            </button>
            <button
              onClick={() => setLearners(generateMockLearners(50))}
              className="p-2 hover:bg-gray-800 rounded-lg text-gray-400 hover:text-white transition-colors"
              title="Refresh data"
            >
              <RefreshCw className="w-5 h-5" />
            </button>
          </div>
        </div>

        {/* Stats */}
        <StatsBar stats={stats} />

        {/* Toolbar */}
        <Toolbar
          searchQuery={searchQuery}
          onSearchChange={setSearchQuery}
          statusFilter={statusFilter}
          onStatusChange={setStatusFilter}
          countryFilter={countryFilter}
          onCountryChange={setCountryFilter}
          countries={countries}
          selectedCount={selectedIds.size}
          onMessageClick={() => setIsComposeOpen(true)}
          onExportClick={handleExport}
          onClearSelection={handleClearSelection}
        />

        {/* Table */}
        <div className="bg-gray-800/30 rounded-xl border border-gray-700/50">
          <LearnersTable
            learners={filteredLearners}
            selectedIds={selectedIds}
            onSelect={handleSelect}
            onSelectAll={handleSelectAll}
            onRowClick={setSelectedLearner}
          />
        </div>

        {/* Results count */}
        <div className="text-sm text-gray-400">
          Showing {filteredLearners.length} of {learners.length} learners
        </div>
      </div>

      {/* Detail Panel */}
      {selectedLearner && (
        <LearnerDetailPanel
          learner={selectedLearner}
          onClose={() => setSelectedLearner(null)}
          onSendEmail={() => {
            setSelectedIds(new Set([selectedLearner.id]));
            setIsComposeOpen(true);
          }}
          onSendPush={() => {
            setSelectedIds(new Set([selectedLearner.id]));
            setIsComposeOpen(true);
          }}
        />
      )}

      {/* Compose Modal */}
      {isComposeOpen && selectedLearners.length > 0 && (
        <ComposeMessageModal
          recipients={selectedLearners}
          onClose={() => setIsComposeOpen(false)}
          onSend={handleSendMessage}
        />
      )}
    </div>
  );
}
