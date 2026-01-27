'use client';

import { Search, Mail, Download, X } from 'lucide-react';

interface ToolbarProps {
  searchQuery: string;
  onSearchChange: (query: string) => void;
  statusFilter: string;
  onStatusChange: (status: string) => void;
  countryFilter: string;
  onCountryChange: (country: string) => void;
  countries: { code: string; flag: string }[];
  selectedCount: number;
  onMessageClick: () => void;
  onExportClick: () => void;
  onClearSelection: () => void;
}

export function Toolbar({
  searchQuery,
  onSearchChange,
  statusFilter,
  onStatusChange,
  countryFilter,
  onCountryChange,
  countries,
  selectedCount,
  onMessageClick,
  onExportClick,
  onClearSelection,
}: ToolbarProps) {
  return (
    <div className="space-y-3">
      <div className="flex flex-wrap gap-3">
        {/* Search */}
        <div className="relative flex-1 min-w-[200px]">
          <Search className="absolute left-3 top-1/2 -translate-y-1/2 w-4 h-4 text-gray-400" />
          <input
            type="text"
            placeholder="Search by name, email, city, country..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            className="w-full pl-10 pr-4 py-2 bg-gray-800/50 border border-gray-700/50 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
          />
        </div>
        
        {/* Status Filter */}
        <select
          value={statusFilter}
          onChange={(e) => onStatusChange(e.target.value)}
          aria-label="Filter by status"
          className="px-4 py-2 bg-gray-800/50 border border-gray-700/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
        >
          <option value="all">All Status</option>
          <option value="free">Free</option>
          <option value="trial">Trial</option>
          <option value="active">Active</option>
          <option value="canceled">Canceled</option>
          <option value="past_due">Past Due</option>
        </select>
        
        {/* Country Filter */}
        <select
          value={countryFilter}
          onChange={(e) => onCountryChange(e.target.value)}
          aria-label="Filter by country"
          className="px-4 py-2 bg-gray-800/50 border border-gray-700/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
        >
          <option value="all">All Countries</option>
          {countries.map((c) => (
            <option key={c.code} value={c.code}>
              {c.flag} {c.code}
            </option>
          ))}
        </select>
      </div>
      
      {/* Bulk Actions */}
      {selectedCount > 0 && (
        <div className="flex items-center gap-3 p-3 bg-blue-500/10 border border-blue-500/20 rounded-lg">
          <span className="text-blue-400 font-medium">{selectedCount} selected</span>
          <button
            onClick={onMessageClick}
            className="flex items-center gap-2 px-3 py-1.5 bg-blue-500 hover:bg-blue-600 text-white rounded-lg text-sm transition-colors"
          >
            <Mail className="w-4 h-4" />
            Message
          </button>
          <button
            onClick={onExportClick}
            className="flex items-center gap-2 px-3 py-1.5 bg-gray-700 hover:bg-gray-600 text-white rounded-lg text-sm transition-colors"
          >
            <Download className="w-4 h-4" />
            Export
          </button>
          <button
            onClick={onClearSelection}
            aria-label="Clear selection"
            className="p-1.5 hover:bg-gray-700 rounded text-gray-400 hover:text-white transition-colors"
          >
            <X className="w-4 h-4" />
          </button>
        </div>
      )}
    </div>
  );
}
