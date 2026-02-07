'use client';

import { Download } from 'lucide-react';
import Link from 'next/link';

export default function JoinPage() {
  const documents = [
    {
      id: 'one-pager',
      title: 'One Pager',
      subtitle: 'Executive Summary',
      description: 'The problem, solution, projections, and ask—on one page.',
      pages: '1 page',
      pdf: '/docs/LOTD_One_Pager.pdf',
    },
    {
      id: 'laguna-ridge',
      title: 'Laguna Ridge',
      subtitle: 'Property Overview',
      description: 'One-page overview of the opportunity, vision, and invitation to participate.',
      pages: '1 page',
      pdf: '/docs/Laguna_Ridge.pdf',
    },
    {
      id: 'investor-memo',
      title: 'Master Investment Thesis',
      subtitle: 'Complete Investment Thesis',
      description: 'Track record, building, product, economics, coalition structure, and risk factors.',
      pages: '45 pages',
      pdf: '/docs/LOTD_Master_Investment_Thesis.pdf',
    },

    {
      id: 'ziggurat',
      title: 'The Ziggurat Transformation',
      subtitle: 'Floor-by-Floor Renovation Plan',
      description: 'Detailed plan for converting one million square feet into manufacturing and broadcast infrastructure.',
      pages: '12 pages',
      pdf: '/docs/Ziggurat_Transformation_Plan.pdf',
    },
  ];

  return (
    <div className="min-h-screen bg-white">
      {/* Navigation */}
      <nav className="fixed top-0 left-0 right-0 z-50 bg-white/80 backdrop-blur-md border-b border-neutral-200">
        <div className="max-w-5xl mx-auto px-6 py-4 flex items-center justify-between">
          <Link href="/" className="flex items-center gap-2">
            <span className="text-neutral-400 text-sm">Curious Kelly</span>
            <span className="text-neutral-300">|</span>
            <span className="font-serif text-neutral-900">The Daily Lesson</span>
          </Link>
          <div className="flex items-center gap-6 text-sm">
            <Link href="/about" className="text-neutral-500 hover:text-neutral-900 transition-colors">About</Link>
            <Link href="/strategic" className="text-neutral-500 hover:text-neutral-900 transition-colors">Strategic</Link>
            <Link href="/join" className="text-neutral-900 font-medium">Investors</Link>
            <span className="flex items-center gap-1.5 px-2 py-0.5 bg-emerald-500/10 border border-emerald-500/20 rounded-full text-[10px] text-emerald-600 font-medium uppercase">
              <span className="w-1.5 h-1.5 rounded-full bg-emerald-500 animate-pulse" />
              Beta
            </span>
          </div>
        </div>
      </nav>

      {/* Hero */}
      <section className="pt-32 pb-20 px-6 border-b border-neutral-200">
        <div className="max-w-3xl mx-auto">
          <p className="text-xs uppercase tracking-[0.3em] text-neutral-400 mb-6">
            Lesson of the Day, PBC
          </p>
          <h1 className="font-serif text-5xl md:text-6xl font-light text-neutral-900 mb-8 leading-tight">
            Join the Coalition
          </h1>
          <p className="text-xl text-neutral-600 leading-relaxed max-w-2xl">
            We are building the infrastructure to deliver quality education to eight billion learners. 
            This is an invitation to participate.
          </p>
        </div>
      </section>

      {/* The Opportunity - Brief */}
      <section className="py-20 px-6 bg-neutral-50">
        <div className="max-w-3xl mx-auto">
          <p className="font-serif text-2xl md:text-3xl text-neutral-800 leading-relaxed">
            The Ziggurat—ninety-two acres, one million square feet, four miles from the Pacific—was 
            designed as the world's largest electronics manufacturing plant. It has never been used 
            for that purpose. <em className="text-neutral-500">Until now.</em>
          </p>
        </div>
      </section>

      {/* Documents */}
      <section className="py-20 px-6 border-b border-neutral-200">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-xs uppercase tracking-[0.3em] text-neutral-400 mb-12">
            Documentation
          </h2>
          
          <div className="space-y-0 divide-y divide-neutral-200">
            {documents.map((doc) => (
              <div
                key={doc.id}
                className="py-8 group"
              >
                <div className="flex flex-col md:flex-row md:items-start md:justify-between gap-6">
                  <div className="flex-1">
                    <div className="flex items-baseline gap-4 mb-2">
                      <h3 className="font-serif text-2xl text-neutral-900">
                        {doc.title}
                      </h3>
                      <span className="text-xs text-neutral-400 uppercase tracking-wider">
                        {doc.pages}
                      </span>
                    </div>
                    <p className="text-sm text-neutral-500 mb-1">
                      {doc.subtitle}
                    </p>
                    <p className="text-neutral-600 max-w-lg">
                      {doc.description}
                    </p>
                  </div>
                  
                  <a
                    href={doc.pdf}
                    target="_blank"
                    rel="noopener noreferrer"
                    className="inline-flex items-center gap-2 text-sm text-neutral-900 hover:text-neutral-600 transition-colors border border-neutral-200 px-4 py-2 rounded-md hover:bg-neutral-50"
                  >
                    <Download className="w-4 h-4" />
                    Download PDF
                  </a>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* How to Participate */}
      <section className="py-20 px-6">
        <div className="max-w-4xl mx-auto">
          <h2 className="text-xs uppercase tracking-[0.3em] text-neutral-400 mb-12">
            How to Participate
          </h2>
          
          <div className="grid md:grid-cols-2 gap-x-16 gap-y-12">
            <div>
              <h3 className="font-serif text-xl text-neutral-900 mb-3">
                Contribute
              </h3>
              <p className="text-neutral-600 leading-relaxed">
                Direct contribution to Lesson of the Day, PBC. Contributors receive permanent 
                recognition, priority access, and a direct relationship through the acquisition.
              </p>
            </div>
            
            <div>
              <h3 className="font-serif text-xl text-neutral-900 mb-3">
                Introduce
              </h3>
              <p className="text-neutral-600 leading-relaxed">
                Connections that accelerate: GSA leadership, LA 2028 Olympics organizing committee, 
                Apple Education, manufacturing partners, infrastructure investors.
              </p>
            </div>
            
            <div>
              <h3 className="font-serif text-xl text-neutral-900 mb-3">
                Support
              </h3>
              <p className="text-neutral-600 leading-relaxed">
                Letters and endorsements strengthen the GSA submission: letter of support, 
                statement of intent for partnership, advisory commitment.
              </p>
            </div>
            
            <div>
              <h3 className="font-serif text-xl text-neutral-900 mb-3">
                Backstop
              </h3>
              <p className="text-neutral-600 leading-relaxed">
                Guarantee acquisition financing. Two hundred million secures the property. 
                Two billion secures the Olympics vision. Three billion builds the century asset.
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Wire Instructions */}
      <section className="py-20 px-6 bg-neutral-900 text-white">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-xs uppercase tracking-[0.3em] text-neutral-500 mb-12">
            Wire Instructions
          </h2>
          
          <div className="grid md:grid-cols-2 gap-12">
            <div className="space-y-6">
              <div>
                <p className="text-xs uppercase tracking-wider text-neutral-500 mb-1">Bank</p>
                <p className="font-serif text-lg">Wells Fargo</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wider text-neutral-500 mb-1">Routing</p>
                <p className="font-mono text-lg tracking-wide">121000248</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wider text-neutral-500 mb-1">Account</p>
                <p className="font-mono text-lg tracking-wide">6035509675</p>
              </div>
            </div>
            
            <div className="space-y-6">
              <div>
                <p className="text-xs uppercase tracking-wider text-neutral-500 mb-1">Beneficiary</p>
                <p className="font-serif text-lg">Lesson of the Day, PBC</p>
              </div>
              <div>
                <p className="text-xs uppercase tracking-wider text-neutral-500 mb-1">Reference</p>
                <p className="font-mono text-lg tracking-wide">LAGUNA RIDGE</p>
              </div>
            </div>
          </div>
        </div>
      </section>

      {/* Contact */}
      <section className="py-20 px-6 border-b border-neutral-200">
        <div className="max-w-3xl mx-auto">
          <h2 className="text-xs uppercase tracking-[0.3em] text-neutral-400 mb-12">
            Contact
          </h2>
          
          <div className="flex flex-col md:flex-row md:justify-between gap-8">
            <div>
              <h3 className="font-serif text-2xl text-neutral-900 mb-1">
                Nicolette Rankin
              </h3>
              <p className="text-neutral-500 mb-6">
                Founder & Chief Executive Officer
              </p>
              
              <div className="space-y-2 text-neutral-600">
                <p>
                  <a href="tel:805-233-5084" className="hover:text-neutral-900 transition-colors">
                    805 233 5084
                  </a>
                </p>
                <p>
                  <a href="mailto:nicolette@thedailylesson.com" className="hover:text-neutral-900 transition-colors">
                    nicolette@thedailylesson.com
                  </a>
                </p>
                <p>
                  <a 
                    href="https://linkedin.com/in/rankinnicolette" 
                    target="_blank" 
                    rel="noopener noreferrer"
                    className="hover:text-neutral-900 transition-colors"
                  >
                    LinkedIn
                  </a>
                </p>
              </div>
            </div>
            
            <div className="md:text-right">
              <p className="text-xs uppercase tracking-[0.2em] text-neutral-400 mb-2">
                California Public Benefit Corporation
              </p>
              <p className="font-mono text-neutral-600">
                Entity No. 5774402
              </p>
            </div>
          </div>
        </div>
      </section>

      {/* Footer */}
      <footer className="py-12 px-6">
        <div className="max-w-3xl mx-auto text-center">
          <p className="font-serif text-lg text-neutral-400 italic">
            Kelly teaches the world.
          </p>
        </div>
      </footer>
    </div>
  );
}
