// components/pricing/faq-section.tsx
'use client'

import { useState } from 'react'
import { ChevronDown } from 'lucide-react'

interface FAQItem {
  question: string
  answer: string
}

const faqs: FAQItem[] = [
  {
    question: "What's included in each tier?",
    answer: "Explorer (Free): Today's lesson only, single language, basic avatar. Learner: 365 lessons, 12+ languages, all avatars, offline downloads. Family: Same as Learner plus up to 6 family members with individual profiles. School: Custom per-district pricing with admin dashboard.",
  },
  {
    question: 'Can I change plans?',
    answer: 'Yes! You can upgrade or downgrade anytime. Changes take effect on your next billing cycle. Downgrading may result in a prorated refund.',
  },
  {
    question: 'Is there a free trial?',
    answer: 'Explorer tier is permanently free - no trial needed. Start with daily lessons and upgrade whenever you\'re ready.',
  },
  {
    question: 'How do I cancel?',
    answer: 'Cancel anytime from your account settings. Your access continues through the end of your billing period. No cancellation fees.',
  },
  {
    question: 'What payment methods do you accept?',
    answer: 'We accept all major credit cards (Visa, Mastercard, American Express) and Apple Pay/Google Pay via Stripe.',
  },
  {
    question: 'Is my payment information secure?',
    answer: 'Yes! We use Stripe for payment processing, which is PCI-DSS Level 1 certified. We never store your card details.',
  },
  {
    question: 'How do I use an affiliate link?',
    answer: 'If you were referred by an affiliate, their ref code should be in the URL (?ref=CODE). Your referrer earns 25-35% commission on your subscription.',
  },
  {
    question: 'Can I switch between monthly and yearly?',
    answer: 'Absolutely. You can change your billing interval anytime. We\'ll adjust your next charge accordingly.',
  },
]

export function FAQSection() {
  const [openIndex, setOpenIndex] = useState<number | null>(null)

  return (
    <div className="border-t border-white/10 px-6 py-16">
      <div className="mx-auto max-w-2xl">
        <h2 className="text-3xl font-bold text-white mb-12">Frequently Asked Questions</h2>

        <div className="space-y-4">
          {faqs.map((faq, index) => (
            <div key={index} className="border border-white/10 rounded-lg overflow-hidden">
              <button
                onClick={() => setOpenIndex(openIndex === index ? null : index)}
                className="w-full px-6 py-4 flex items-center justify-between hover:bg-white/5 transition-colors text-left"
              >
                <span className="font-medium text-white">{faq.question}</span>
                <ChevronDown
                  className={`w-5 h-5 text-white/60 transition-transform ${
                    openIndex === index ? 'rotate-180' : ''
                  }`}
                />
              </button>

              {openIndex === index && (
                <div className="px-6 py-4 bg-white/5 border-t border-white/10">
                  <p className="text-white/70 leading-relaxed">{faq.answer}</p>
                </div>
              )}
            </div>
          ))}
        </div>
      </div>
    </div>
  )
}
