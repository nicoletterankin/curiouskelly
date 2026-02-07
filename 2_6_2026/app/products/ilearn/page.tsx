// app/products/ilearn/page.tsx
// CORRECTED — 5 Device Lineup
import { ILEARN_DEVICES, ILEARN_BUNDLES } from '@/lib/products';
import { Check, Zap, Shield, Globe, Sparkles, ArrowLeft } from 'lucide-react';
import Link from 'next/link';

export default function ILearnPage() {
  const devices = Object.values(ILEARN_DEVICES);
  
  return (
    <div className="min-h-screen bg-white">
      {/* Header */}
      <header className="sticky top-0 z-50 bg-white/90 backdrop-blur-xl border-b border-gray-200">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <Link 
              href="/products" 
              className="p-2 rounded-lg hover:bg-gray-100 transition"
            >
              <ArrowLeft className="w-5 h-5 text-gray-600" />
            </Link>
            <div>
              <h1 className="text-xl font-semibold text-gray-900">iLearn Devices</h1>
              <p className="text-xs text-gray-500">Built for learning. Nothing else.</p>
            </div>
          </div>
          <Link
            href="/coalition"
            className="px-4 py-2 bg-blue-600 text-white text-sm font-medium rounded-lg hover:bg-blue-500 transition"
          >
            Invest in iLearn
          </Link>
        </div>
      </header>

      {/* Hero */}
      <section className="relative bg-gradient-to-b from-gray-900 to-gray-800 text-white py-24">
        <div className="max-w-7xl mx-auto px-6">
          <p className="text-blue-400 font-semibold">iLearn Hardware</p>
          <h1 className="mt-2 text-5xl md:text-6xl font-bold tracking-tight">
            5 devices for<br />8 billion humans
          </h1>
          <p className="mt-6 text-xl text-gray-300 max-w-2xl">
            From the $10 Nano for villages without electricity to the $4,999 Ultra for institutions. 
            Everyone gets Kelly. No one is left behind.
          </p>
          
          {/* Quick price ladder */}
          <div className="mt-12 flex flex-wrap gap-4">
            {devices.map((device) => (
              <div key={device.id} className={`px-4 py-2 rounded-full bg-gradient-to-r ${device.color} text-white text-sm font-medium`}>
                {device.name}: ${device.price.toLocaleString()}
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* The Philosophy */}
      <section className="py-16 bg-gray-50">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-3xl font-bold">One principle: No barriers to learning</h2>
          <p className="mt-4 text-xl text-gray-600">
            Every device exists to reach a different segment of humanity. 
            The premium devices fund the mission devices.
          </p>
          
          <div className="mt-12 grid md:grid-cols-3 gap-8">
            <div className="p-6">
              <div className="w-12 h-12 mx-auto bg-green-100 rounded-full flex items-center justify-center">
                <Globe className="w-6 h-6 text-green-600" />
              </div>
              <h3 className="mt-4 font-bold">Nano + Mini</h3>
              <p className="mt-2 text-gray-600">Reach the 3B without resources</p>
            </div>
            <div className="p-6">
              <div className="w-12 h-12 mx-auto bg-blue-100 rounded-full flex items-center justify-center">
                <Zap className="w-6 h-6 text-blue-600" />
              </div>
              <h3 className="mt-4 font-bold">Tiny + Pro</h3>
              <p className="mt-2 text-gray-600">Serve students and professionals</p>
            </div>
            <div className="p-6">
              <div className="w-12 h-12 mx-auto bg-amber-100 rounded-full flex items-center justify-center">
                <Sparkles className="w-6 h-6 text-amber-600" />
              </div>
              <h3 className="mt-4 font-bold">Ultra</h3>
              <p className="mt-2 text-gray-600">Fund 160 Nanos per unit sold</p>
            </div>
          </div>
        </div>
      </section>

      {/* Device Grid */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-center">Choose your iLearn</h2>
          
          <div className="mt-16 space-y-12">
            {devices.map((device) => (
              <div 
                key={device.id}
                className={`rounded-3xl overflow-hidden ${
                  device.premium ? 'ring-4 ring-amber-400 shadow-2xl' : 
                  device.featured ? 'ring-2 ring-blue-500 shadow-lg' : 
                  'border border-gray-200'
                }`}
              >
                <div className={`bg-gradient-to-r ${device.color} p-8 md:p-12`}>
                  <div className="grid md:grid-cols-2 gap-8 items-center">
                    {/* Content */}
                    <div className="text-white">
                      {device.premium && (
                        <span className="inline-block px-3 py-1 bg-amber-400 text-amber-900 text-xs font-bold rounded-full mb-4">
                          FLAGSHIP - FUNDS THE MISSION
                        </span>
                      )}
                      {device.featured && !device.premium && (
                        <span className="inline-block px-3 py-1 bg-white/20 text-white text-xs font-bold rounded-full mb-4">
                          MOST POPULAR
                        </span>
                      )}
                      
                      <p className="text-sm opacity-80">{device.ageRange}</p>
                      <h3 className="mt-1 text-4xl font-bold">{device.name}</h3>
                      <p className="mt-2 text-xl opacity-90">{device.tagline}</p>
                      
                      <div className="mt-6 flex items-baseline gap-2">
                        <span className="text-5xl font-bold">${device.price.toLocaleString()}</span>
                        {device.price >= 100 && (
                          <span className="text-lg opacity-70">
                            or ${Math.round(device.price / 12)}/mo
                          </span>
                        )}
                      </div>
                      
                      <p className="mt-6 text-sm opacity-80">{device.forWhom}</p>
                      
                      <button className={`mt-8 px-8 py-4 rounded-xl font-semibold transition ${
                        device.premium 
                          ? 'bg-amber-400 text-amber-900 hover:bg-amber-300' 
                          : 'bg-white text-gray-900 hover:bg-gray-100'
                      }`}>
                        {device.channels?.includes('Direct') ? 'Buy Now' : 'Request Quote'}
                      </button>
                    </div>
                    
                    {/* Device visualization */}
                    <div className="flex justify-center">
                      <div className={`bg-white/10 backdrop-blur rounded-3xl p-8 ${
                        device.premium ? 'w-72 h-96' : 'w-48 h-64'
                      } flex items-center justify-center`}>
                        <span className={device.premium ? 'text-8xl' : 'text-6xl'}>
                          {device.id === 'ilearn-nano' && '📟'}
                          {device.id === 'ilearn-mini' && '📱'}
                          {device.id === 'ilearn-tiny' && '📱'}
                          {device.id === 'ilearn-pro' && '💻'}
                          {device.id === 'ilearn-ultra' && '🖥️'}
                        </span>
                      </div>
                    </div>
                  </div>
                </div>
                
                {/* Specs section */}
                <div className="p-8 md:p-12 bg-white">
                  <div className="grid md:grid-cols-3 gap-8">
                    {/* Specs */}
                    <div>
                      <h4 className="font-bold text-gray-900">Specifications</h4>
                      <ul className="mt-4 space-y-2">
                        {(device.specs || []).map((spec) => (
                          <li key={spec} className="flex items-start gap-2 text-sm text-gray-600">
                            <Check className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                            <span>{spec}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    {/* In the box */}
                    <div>
                      <h4 className="font-bold text-gray-900">In the box</h4>
                      <ul className="mt-4 space-y-2">
                        {(device.inTheBox || []).map((item) => (
                          <li key={item} className="flex items-start gap-2 text-sm text-gray-600">
                            <span className="text-blue-500">-</span>
                            <span>{item}</span>
                          </li>
                        ))}
                      </ul>
                    </div>
                    
                    {/* Why + Success */}
                    <div>
                      <h4 className="font-bold text-gray-900">Why &ldquo;{device.name.split(' ')[1]}&rdquo;?</h4>
                      <p className="mt-2 text-sm text-gray-600">{device.whyThisName}</p>
                      
                      <h4 className="mt-6 font-bold text-gray-900">Success looks like</h4>
                      <p className="mt-2 text-sm text-gray-600 italic">&ldquo;{device.successLooksLike}&rdquo;</p>
                      
                      {device.crossSubsidy && (
                        <div className="mt-6 p-4 bg-amber-50 rounded-xl">
                          <p className="text-sm font-bold text-amber-800">
                            Each Ultra sold = {device.crossSubsidy.nanosSubsidized} Nanos deployed
                          </p>
                          <p className="text-xs text-amber-600 mt-1">
                            ${device.crossSubsidy.profitPerUnit.toLocaleString()} gross profit per unit
                          </p>
                        </div>
                      )}
                    </div>
                  </div>
                  
                  {/* Channels */}
                  <div className="mt-8 pt-8 border-t border-gray-100">
                    <p className="text-sm text-gray-500">
                      Available through: {device.channels?.join(' - ') || 'Contact us'}
                    </p>
                  </div>
                </div>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Comparison Table */}
      <section className="py-24 bg-gray-50">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-center">Compare all devices</h2>
          
          <div className="mt-12 overflow-x-auto">
            <table className="w-full min-w-[800px]">
              <thead>
                <tr className="border-b-2 border-gray-200">
                  <th className="py-4 px-4 text-left">Device</th>
                  <th className="py-4 px-4 text-left">For</th>
                  <th className="py-4 px-4 text-right">Price</th>
                  <th className="py-4 px-4 text-center">Display</th>
                  <th className="py-4 px-4 text-center">Battery</th>
                  <th className="py-4 px-4 text-center">Internet</th>
                </tr>
              </thead>
              <tbody>
                <tr className="border-b border-gray-100 bg-green-50">
                  <td className="py-4 px-4 font-bold">Nano</td>
                  <td className="py-4 px-4 text-sm text-gray-600">3B without resources</td>
                  <td className="py-4 px-4 text-right font-mono font-bold">$10</td>
                  <td className="py-4 px-4 text-center text-sm">4&rdquo; e-ink</td>
                  <td className="py-4 px-4 text-center text-sm">Solar/crank</td>
                  <td className="py-4 px-4 text-center text-sm">None needed</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-4 px-4 font-bold">Mini</td>
                  <td className="py-4 px-4 text-sm text-gray-600">Ages 2-7</td>
                  <td className="py-4 px-4 text-right font-mono font-bold">$99</td>
                  <td className="py-4 px-4 text-center text-sm">7&rdquo; color</td>
                  <td className="py-4 px-4 text-center text-sm">12 hours</td>
                  <td className="py-4 px-4 text-center text-sm">Optional</td>
                </tr>
                <tr className="border-b border-gray-100 bg-blue-50">
                  <td className="py-4 px-4 font-bold">Tiny *</td>
                  <td className="py-4 px-4 text-sm text-gray-600">Ages 8-18</td>
                  <td className="py-4 px-4 text-right font-mono font-bold">$249</td>
                  <td className="py-4 px-4 text-center text-sm">8&rdquo; + stylus</td>
                  <td className="py-4 px-4 text-center text-sm">16 hours</td>
                  <td className="py-4 px-4 text-center text-sm">Online/offline</td>
                </tr>
                <tr className="border-b border-gray-100">
                  <td className="py-4 px-4 font-bold">Pro</td>
                  <td className="py-4 px-4 text-sm text-gray-600">Ages 16+</td>
                  <td className="py-4 px-4 text-right font-mono font-bold">$749</td>
                  <td className="py-4 px-4 text-center text-sm">10&rdquo; + keyboard</td>
                  <td className="py-4 px-4 text-center text-sm">20 hours</td>
                  <td className="py-4 px-4 text-center text-sm">Full connectivity</td>
                </tr>
                <tr className="border-b border-gray-100 bg-amber-50">
                  <td className="py-4 px-4 font-bold">Ultra **</td>
                  <td className="py-4 px-4 text-sm text-gray-600">Institutions</td>
                  <td className="py-4 px-4 text-right font-mono font-bold">$4,999</td>
                  <td className="py-4 px-4 text-center text-sm">15&rdquo; 4K OLED</td>
                  <td className="py-4 px-4 text-center text-sm">All day</td>
                  <td className="py-4 px-4 text-center text-sm">On-device AI</td>
                </tr>
              </tbody>
            </table>
            <p className="mt-4 text-sm text-gray-500 text-center">* Most Popular | ** Flagship - Funds the Mission</p>
          </div>
        </div>
      </section>

      {/* Education Bundles */}
      <section className="py-24">
        <div className="max-w-7xl mx-auto px-6">
          <h2 className="text-3xl font-bold text-center">Education Bundles</h2>
          <p className="mt-4 text-xl text-gray-600 text-center">
            Volume pricing for schools and institutions
          </p>
          
          <div className="mt-16 grid md:grid-cols-2 lg:grid-cols-3 gap-8">
            {Object.values(ILEARN_BUNDLES).map((bundle) => (
              <div 
                key={bundle.id}
                className={`bg-white rounded-2xl p-8 border-2 ${
                  bundle.featured ? 'border-blue-500 shadow-lg' : 'border-gray-200'
                }`}
              >
                {bundle.featured && (
                  <span className="inline-block px-3 py-1 bg-blue-100 text-blue-700 text-xs font-bold rounded-full mb-4">
                    BEST VALUE
                  </span>
                )}
                
                <h3 className="text-xl font-bold">{bundle.name}</h3>
                <p className="text-gray-500">{bundle.tagline}</p>
                
                <div className="mt-6 flex items-baseline justify-between">
                  <span className="text-3xl font-bold">${bundle.price.toLocaleString()}</span>
                  <span className="text-green-600 font-medium">Save {bundle.savingsPercent}%</span>
                </div>
                
                <p className="mt-2 text-sm text-gray-500">
                  ${bundle.perUnit}/unit
                </p>
                
                <ul className="mt-6 space-y-2">
                  {bundle.contents.map((item) => (
                    <li key={item} className="flex items-start gap-2 text-sm text-gray-600">
                      <Check className="w-4 h-4 text-green-500 mt-0.5 flex-shrink-0" />
                      <span>{item}</span>
                    </li>
                  ))}
                </ul>
                
                <button className="mt-8 w-full py-3 bg-blue-600 text-white rounded-xl font-semibold hover:bg-blue-700 transition">
                  Request Quote
                </button>
              </div>
            ))}
          </div>
        </div>
      </section>

      {/* Cross-Subsidy Explainer */}
      <section className="py-24 bg-gray-900 text-white">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold">The Ultra funds the mission</h2>
          <p className="mt-6 text-xl text-gray-300">
            Every iLearn Ultra sold generates $3,199 in gross profit. 
            That&apos;s enough to deploy 160 iLearn Nano devices to learners 
            who couldn&apos;t otherwise access Kelly.
          </p>
          
          <div className="mt-12 grid md:grid-cols-3 gap-8">
            <div className="p-6 bg-gray-800 rounded-2xl">
              <p className="text-5xl font-bold text-amber-400">1</p>
              <p className="mt-2 text-gray-300">Ultra sold</p>
            </div>
            <div className="p-6 flex items-center justify-center">
              <span className="text-4xl">=</span>
            </div>
            <div className="p-6 bg-gray-800 rounded-2xl">
              <p className="text-5xl font-bold text-green-400">160</p>
              <p className="mt-2 text-gray-300">Nano devices deployed</p>
            </div>
          </div>
          
          <p className="mt-12 text-gray-400">
            By 2032, Ultra sales will fund 3.2 million Nano deployments.
          </p>
        </div>
      </section>

      {/* CTA */}
      <section className="py-24 bg-blue-600 text-white">
        <div className="max-w-4xl mx-auto px-6 text-center">
          <h2 className="text-4xl font-bold">Ready to learn?</h2>
          <p className="mt-4 text-xl text-blue-100">
            Choose the iLearn that&apos;s right for you.
          </p>
          <div className="mt-8 flex flex-wrap gap-4 justify-center">
            <button className="px-8 py-4 bg-white text-blue-600 rounded-xl font-semibold hover:bg-blue-50 transition">
              Shop Consumer Devices
            </button>
            <button className="px-8 py-4 bg-blue-700 text-white rounded-xl font-semibold hover:bg-blue-800 transition">
              Education Bundles
            </button>
            <Link 
              href="/coalition"
              className="px-8 py-4 bg-amber-400 text-amber-900 rounded-xl font-semibold hover:bg-amber-300 transition"
            >
              Invest in the Mission
            </Link>
          </div>
        </div>
      </section>
    </div>
  );
}
