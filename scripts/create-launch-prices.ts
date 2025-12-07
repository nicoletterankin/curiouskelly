/**
 * Create Launch Pricing in Stripe
 * 
 * Run: npx ts-node scripts/create-launch-prices.ts
 * 
 * This script creates the $4.99/month and $49.99/year "Founding Member" prices
 * that match the checkout page pricing.
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
  console.log('💰 CREATING LAUNCH PRICING');
  console.log('========================================\n');

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
        name: 'Curious Kelly - Founding Member',
        description: 'Daily AI-powered lessons for the whole family. Founding member pricing - locked forever.',
        metadata: {
          type: 'founding_member',
          launch_date: '2025-12-17',
        },
      });
      console.log(`✅ Created product: ${product.id}`);
    } else {
      console.log(`✅ Using existing product: ${product.id} (${product.name})`);
    }

    // Create Monthly Founding Member Price ($4.99/month)
    console.log('\nCreating Monthly Founding Member price ($4.99/month)...');
    const monthlyPrice = await stripe.prices.create({
      product: product.id,
      unit_amount: 499, // $4.99 in cents
      currency: 'usd',
      recurring: {
        interval: 'month',
        trial_period_days: 7,
      },
      nickname: 'Founding Member Monthly',
      metadata: {
        plan_type: 'monthly',
        is_founding_price: 'true',
        regular_price_cents: '999', // Regular price for reference
      },
    });
    console.log(`✅ Created monthly price: ${monthlyPrice.id}`);

    // Create Annual Founding Member Price ($49.99/year)
    console.log('\nCreating Annual Founding Member price ($49.99/year)...');
    const annualPrice = await stripe.prices.create({
      product: product.id,
      unit_amount: 4999, // $49.99 in cents
      currency: 'usd',
      recurring: {
        interval: 'year',
        trial_period_days: 7,
      },
      nickname: 'Founding Member Annual',
      metadata: {
        plan_type: 'annual',
        is_founding_price: 'true',
        regular_price_cents: '9900', // Regular price for reference
        savings: '50%',
      },
    });
    console.log(`✅ Created annual price: ${annualPrice.id}`);

    // Summary
    console.log('\n========================================');
    console.log('📋 UPDATE YOUR .ENV FILE');
    console.log('========================================\n');
    console.log('Add or update these values:\n');
    console.log(`STRIPE_PRICE_MONTHLY_FOUNDING=${monthlyPrice.id}`);
    console.log(`STRIPE_PRICE_ANNUAL_FOUNDING=${annualPrice.id}`);
    console.log('\nOr replace existing prices:');
    console.log(`STRIPE_PRICE_MONTHLY=${monthlyPrice.id}`);
    console.log(`STRIPE_PRICE_ANNUAL=${annualPrice.id}`);

    console.log('\n========================================');
    console.log('✅ LAUNCH PRICES CREATED SUCCESSFULLY');
    console.log('========================================\n');

    return {
      monthly: monthlyPrice.id,
      annual: annualPrice.id,
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

