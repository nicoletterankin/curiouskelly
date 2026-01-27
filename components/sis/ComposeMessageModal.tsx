'use client';

import { useState } from 'react';
import { X, Send } from 'lucide-react';
import { Learner } from '../../lib/sis-types';
import { MESSAGE_TEMPLATES } from '../../lib/sis-mock-data';

interface ComposeMessageModalProps {
  recipients: Learner[];
  onClose: () => void;
  onSend: (subject: string, body: string, type: 'email' | 'push') => void;
}

export function ComposeMessageModal({ recipients, onClose, onSend }: ComposeMessageModalProps) {
  const [templateId, setTemplateId] = useState('');
  const [subject, setSubject] = useState('');
  const [body, setBody] = useState('');
  const [type, setType] = useState<'email' | 'push'>('email');

  const handleTemplateChange = (id: string) => {
    setTemplateId(id);
    const template = MESSAGE_TEMPLATES.find(t => t.id === id);
    if (template) {
      setSubject(template.subject);
      setBody(template.body);
      setType(template.type);
    }
  };

  const insertVariable = (variable: string) => {
    setBody(prev => prev + `{{${variable}}}`);
  };

  const eligibleCount = recipients.filter(r => 
    type === 'email' ? (r.email && r.emailOptIn) : r.pushOptIn
  ).length;

  return (
    <>
      {/* Backdrop */}
      <div className="fixed inset-0 bg-black/50 z-50" onClick={onClose} />
      
      {/* Modal */}
      <div className="fixed top-1/2 left-1/2 -translate-x-1/2 -translate-y-1/2 w-full max-w-lg bg-gray-900 rounded-xl border border-gray-700/50 z-50">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-700/50">
          <div>
            <h2 className="text-lg font-semibold text-white">Compose Message</h2>
            <p className="text-sm text-gray-400">{eligibleCount} of {recipients.length} will receive</p>
          </div>
          <button onClick={onClose} aria-label="Close modal" className="p-2 hover:bg-gray-800 rounded-lg">
            <X className="w-5 h-5 text-gray-400" />
          </button>
        </div>
        
        <div className="p-4 space-y-4">
          {/* Template */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Template</label>
            <select
              value={templateId}
              onChange={(e) => handleTemplateChange(e.target.value)}
              aria-label="Message template"
              className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700/50 rounded-lg text-white focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            >
              <option value="">Select a template...</option>
              {MESSAGE_TEMPLATES.map(t => (
                <option key={t.id} value={t.id}>{t.name} ({t.type})</option>
              ))}
            </select>
          </div>
          
          {/* Type */}
          <div className="flex gap-2">
            <button
              onClick={() => setType('email')}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                type === 'email' ? 'bg-blue-500 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Email
            </button>
            <button
              onClick={() => setType('push')}
              className={`flex-1 py-2 rounded-lg text-sm font-medium transition-colors ${
                type === 'push' ? 'bg-blue-500 text-white' : 'bg-gray-800 text-gray-400 hover:bg-gray-700'
              }`}
            >
              Push Notification
            </button>
          </div>
          
          {/* Subject */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Subject</label>
            <input
              type="text"
              value={subject}
              onChange={(e) => setSubject(e.target.value)}
              placeholder="Enter subject..."
              className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700/50 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50"
            />
          </div>
          
          {/* Body */}
          <div>
            <label className="block text-sm font-medium text-gray-400 mb-1">Message</label>
            <textarea
              value={body}
              onChange={(e) => setBody(e.target.value)}
              placeholder="Enter your message..."
              rows={6}
              className="w-full px-4 py-2 bg-gray-800/50 border border-gray-700/50 rounded-lg text-white placeholder-gray-500 focus:outline-none focus:ring-2 focus:ring-blue-500/50 resize-none"
            />
          </div>
          
          {/* Variables */}
          <div className="flex flex-wrap gap-2">
            <span className="text-sm text-gray-400">Variables:</span>
            {['name', 'email', 'streak', 'lessons', 'day'].map(v => (
              <button
                key={v}
                onClick={() => insertVariable(v)}
                className="px-2 py-1 bg-gray-800 hover:bg-gray-700 text-gray-300 rounded text-xs font-mono transition-colors"
              >
                {`{{${v}}}`}
              </button>
            ))}
          </div>
        </div>
        
        {/* Footer */}
        <div className="flex justify-end gap-3 p-4 border-t border-gray-700/50">
          <button
            onClick={onClose}
            className="px-4 py-2 text-gray-400 hover:text-white transition-colors"
          >
            Cancel
          </button>
          <button
            onClick={() => onSend(subject, body, type)}
            disabled={!subject.trim() || !body.trim() || eligibleCount === 0}
            className="flex items-center gap-2 px-4 py-2 bg-blue-500 hover:bg-blue-600 disabled:bg-gray-700 disabled:text-gray-500 text-white rounded-lg transition-colors"
          >
            <Send className="w-4 h-4" />
            Send Message
          </button>
        </div>
      </div>
    </>
  );
}
