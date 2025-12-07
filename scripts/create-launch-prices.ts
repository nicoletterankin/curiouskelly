/**
 * Create Launch Pricing in Stripe
 * 
 * Run: npx ts-node scripts/create-launch-prices.ts
 * 
 * 🔒 LOCKED PRICING (See PRICING_LOCKED.md):
 * - Monthly: $7.99/month
 * - Annual: $49.99/year (DEFAULT)
 * - Family: $99.99/year
 * - Lifetime: $199.99 one-time
 * - Gifts: $24.99 (3mo), $39.99 (6mo), $49.99 (12mo), $149.99 (lifetime)
 */

import Stripe from 'stripe';
import * as dotenv from 'dotenv';

dotenv.config();

const stripeKey = process.env.STRIPE_SECRET_KEY;

if (!stripeKey) {
  console.error('❌ STRIPE_SECRET_KEY not found in environment');
  process.exit(1);
}

const stripe = new Stripe(stripeKey, {
  apiVersion: '2023-10-16',
});

async function createLaunchPrices() {
  console.log('\n========================================');
  console.log('💰 CREATING LOCKED LAUNCH PRICING');
  console.log('========================================\n');
  console.log('📋 Reference: PRICING_LOCKED.md\n');

  try {
    // First, let's find or create the main product
    const products = await stripe.products.list({ limit: 10 });
    
    // Look for an existing Curious Kelly product
    let product = products.data.find(p => 
      p.name.toLowerCase().includes('curious kelly') || 
      p.name.toLowerCase().includes('monthly') ||
      p.name.toLowerCase().includes('subscription')
    );

    if (!product) {
      console.log('Creating new product...');
      product = await stripe.products.create({
        name: 'Curious Kelly Subscription',
        description: 'Daily AI-powered lessons for the whole family. 365 days of learning with Kelly.',
        metadata: {
          type: 'subscription',
          launch_date: '2025-12-17',
        },
      });
      console.log(`✅ Created product: ${product.id}`);
    } else {
      console.log(`✅ Using existing product: ${product.id} (${product.name})`);
    }

    // Create Monthly Price ($7.99/month) - LOCKED
    console.log('\nCreating Monthly price ($7.99/month)...');
    const monthlyPrice = await stripe.prices.create({
      product: product.id,
      unit_amount: 799, // $7.99 in cents - LOCKED
      currency: 'usd',
      recurring: {
        interval: 'month',
        trial_period_days: 7,
      },
      nickname: 'Monthly',
      metadata: {
        plan_type: 'monthly',
        display_price: '$7.99',
      },
    });
    console.log(`✅ Created monthly price: ${monthlyPrice.id}`);

    // Create Annual Price ($49.99/year) - LOCKED
    console.log('\nCreating Annual price ($49.99/year)...');
    const annualPrice = await stripe.prices.create({
      product: product.id,
      unit_amount: 4999, // $49.99 in cents - LOCKED
      currency: 'usd',
      recurring: {
        interval: 'year',
        trial_period_days: 7,
      },
      nickname: 'Annual - Best Value',
      metadata: {
        plan_type: 'annual',
        display_price: '$49.99',
        monthly_equivalent: '$4.17',
        savings: '48%',
      },
    });
    console.log(`✅ Created annual price: ${annualPrice.id}`);

    // Create Family Price ($99.99/year) - LOCKED
    console.log('\nCreating Family price ($99.99/year)...');
    const familyPrice = await stripe.prices.create({
      product: product.id,
      unit_amount: 9999, // $99.99 in cents - LOCKED
      currency: 'usd',
      recurring: {
        interval: 'year',
        trial_period_days: 7,
      },
      nickname: 'Family - Up to 6 members',
      metadata: {
        plan_type: 'family',
        display_price: '$99.99',
        max_members: '6',
      },
    });
    console.log(`✅ Created family price: ${familyPrice.id}`);

    // Create Lifetime Price ($199.99 one-time) - LOCKED
    console.log('\nCreating Lifetime price ($199.99 one-time)...');
    const lifetimePrice = await stripe.prices.create({
      product: product.id,
      unit_amount: 19999, // $199.99 in cents - LOCKED
      currency: 'usd',
      nickname: 'Lifetime Access',
      metadata: {
        plan_type: 'lifetime',
        display_price: '$199.99',
      },
    });
    console.log(`✅ Created lifetime price: ${lifetimePrice.id}`);

    // Create Gift Prices - LOCKED
    console.log('\nCreating Gift prices...');
    
    const gift3moPrice = await stripe.prices.create({
      product: product.id,
      unit_amount: 2499, // $24.99 - LOCKED
      currency: 'usd',
      nickname: 'Gift 3 Months',
      metadata: { plan_type: 'gift_3mo', display_price: '$24.99', duration_months: '3' },
    });
    console.log(`✅ Created gift 3mo price: ${gift3moPrice.id}`);

    const gift6moPrice = await stripe.prices.create({
      product: product.id,
      unit_amount: 3999, // $39.99 - LOCKED
      currency: 'usd',
      nickname: 'Gift 6 Months',
      metadata: { plan_type: 'gift_6mo', display_price: '$39.99', duration_months: '6' },
    });
    console.log(`✅ Created gift 6mo price: ${gift6moPrice.id}`);

    const gift12moPrice = await stripe.prices.create({
      product: product.id,
      unit_amount: 4999, // $49.99 - LOCKED
      currency: 'usd',
      nickname: 'Gift 12 Months',
      metadata: { plan_type: 'gift_12mo', display_price: '$49.99', duration_months: '12' },
    });
    console.log(`✅ Created gift 12mo price: ${gift12moPrice.id}`);

    const giftLifetimePrice = await stripe.prices.create({
      product: product.id,
      unit_amount: 14999, // $149.99 - LOCKED
      currency: 'usd',
      nickname: 'Gift Lifetime',
      metadata: { plan_type: 'gift_lifetime', display_price: '$149.99' },
    });
    console.log(`✅ Created gift lifetime price: ${giftLifetimePrice.id}`);

    // Summary
    console.log('\n========================================');
    console.log('📋 UPDATE YOUR .ENV FILE');
    console.log('========================================\n');
    console.log('# Subscription Prices (LOCKED)');
    console.log(`STRIPE_PRICE_MONTHLY=${monthlyPrice.id}`);
    console.log(`STRIPE_PRICE_ANNUAL=${annualPrice.id}`);
    console.log(`STRIPE_PRICE_FAMILY=${familyPrice.id}`);
    console.log(`STRIPE_PRICE_LIFETIME=${lifetimePrice.id}`);
    console.log('\n# Gift Prices (LOCKED)');
    console.log(`STRIPE_PRICE_GIFT_3MO=${gift3moPrice.id}`);
    console.log(`STRIPE_PRICE_GIFT_6MO=${gift6moPrice.id}`);
    console.log(`STRIPE_PRICE_GIFT_12MO=${gift12moPrice.id}`);
    console.log(`STRIPE_PRICE_GIFT_LIFETIME=${giftLifetimePrice.id}`);

    console.log('\n========================================');
    console.log('✅ ALL LOCKED PRICES CREATED');
    console.log('========================================\n');
    console.log('📋 Pricing Reference: PRICING_LOCKED.md');
    console.log('');
    console.log('💰 LOCKED PRICING SUMMARY:');
    console.log('   Monthly:  $7.99/mo');
    console.log('   Annual:   $49.99/yr (DEFAULT)');
    console.log('   Family:   $99.99/yr');
    console.log('   Lifetime: $199.99');
    console.log('   Gifts:    $24.99 / $39.99 / $49.99 / $149.99');
    console.log('');

    return {
      monthly: monthlyPrice.id,
      annual: annualPrice.id,
      family: familyPrice.id,
      lifetime: lifetimePrice.id,
      gift_3mo: gift3moPrice.id,
      gift_6mo: gift6moPrice.id,
      gift_12mo: gift12moPrice.id,
      gift_lifetime: giftLifetimePrice.id,
    };

  } catch (error) {
    console.error('❌ Error creating prices:', error);
    throw error;
  }
}

// Also create a function to list current prices
async function listCurrentPrices() {
  console.log('\n========================================');
  console.log('📊 CURRENT STRIPE PRICES');
  console.log('========================================\n');

  const prices = await stripe.prices.list({ 
    limit: 20,
    active: true,
    expand: ['data.product'],
  });

  console.log('Active prices:\n');
  for (const price of prices.data) {
    const productName = typeof price.product === 'object' ? price.product.name : price.product;
    const interval = price.recurring?.interval || 'one-time';
    const amount = (price.unit_amount || 0) / 100;
    
    console.log(`  ${price.id}`);
    console.log(`    Product: ${productName}`);
    console.log(`    Amount: $${amount.toFixed(2)}/${interval}`);
    console.log(`    Nickname: ${price.nickname || 'N/A'}`);
    console.log('');
  }
}

// Main execution
async function main() {
  const args = process.argv.slice(2);
  
  if (args.includes('--list')) {
    await listCurrentPrices();
  } else if (args.includes('--create')) {
    await createLaunchPrices();
  } else {
    console.log('Usage:');
    console.log('  npx ts-node scripts/create-launch-prices.ts --list    # List current prices');
    console.log('  npx ts-node scripts/create-launch-prices.ts --create  # Create launch prices');
  }
}

main().catch(console.error);

