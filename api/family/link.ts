/**
 * Family Account Link API
 * 
 * Allows a parent to link a child's account to their family.
 * This enables the parent to manage the child's earnings and
 * provides COPPA compliance for users under 13.
 */

import type { VercelRequest, VercelResponse } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

interface LinkRequest {
  childEmail: string;
  childCode?: string; // Optional verification code sent to child
}

interface ApiResponse {
  success: boolean;
  message: string;
  childId?: string;
  childName?: string;
}

export default async function handler(
  req: VercelRequest,
  res: VercelResponse
): Promise<void> {
  res.setHeader('Access-Control-Allow-Origin', '*');
  res.setHeader('Access-Control-Allow-Methods', 'POST, OPTIONS');
  res.setHeader('Access-Control-Allow-Headers', 'Authorization, Content-Type');

  if (req.method === 'OPTIONS') {
    res.status(200).end();
    return;
  }

  if (req.method !== 'POST') {
    res.status(405).json({ success: false, message: 'Method not allowed' });
    return;
  }

  const authHeader = req.headers.authorization;
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    res.status(401).json({ success: false, message: 'Unauthorized' });
    return;
  }

  const token = authHeader.replace('Bearer ', '');
  const { childEmail } = req.body as LinkRequest;

  if (!childEmail) {
    res.status(400).json({ success: false, message: 'Child email is required' });
    return;
  }

  const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
  const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;
  const supabaseAnonKey = process.env.PUBLIC_SUPABASE_ANON_KEY || process.env.NEXT_PUBLIC_SUPABASE_ANON_KEY;

  if (!supabaseUrl || !supabaseServiceKey || !supabaseAnonKey) {
    res.status(500).json({ success: false, message: 'Server configuration error' });
    return;
  }

  const supabaseAuth = createClient(supabaseUrl, supabaseAnonKey);
  const supabaseAdmin = createClient(supabaseUrl, supabaseServiceKey);

  try {
    // Get parent user from token
    const { data: { user: parentUser }, error: authError } = await supabaseAuth.auth.getUser(token);
    
    if (authError || !parentUser) {
      res.status(401).json({ success: false, message: 'Invalid token' });
      return;
    }

    // Get parent's profile
    const { data: parentProfile, error: parentError } = await supabaseAdmin
      .from('users')
      .select('id, email, calculated_age, is_family_admin')
      .eq('id', parentUser.id)
      .single();

    if (parentError || !parentProfile) {
      res.status(404).json({ success: false, message: 'Parent profile not found' });
      return;
    }

    // Verify parent is an adult (18+)
    // Use the view to get calculated age
    const { data: parentAgeData } = await supabaseAdmin
      .from('users_with_age')
      .select('calculated_age')
      .eq('id', parentUser.id)
      .single();

    const parentAge = parentAgeData?.calculated_age;
    if (parentAge !== null && parentAge < 18) {
      res.status(403).json({ 
        success: false, 
        message: 'Only adults (18+) can be family administrators' 
      });
      return;
    }

    // Find child by email
    const { data: childProfile, error: childError } = await supabaseAdmin
      .from('users')
      .select('id, email, display_name, parent_account_id')
      .eq('email', childEmail.toLowerCase().trim())
      .single();

    if (childError || !childProfile) {
      res.status(404).json({ 
        success: false, 
        message: 'No account found with that email' 
      });
      return;
    }

    // Check if child already has a parent
    if (childProfile.parent_account_id) {
      if (childProfile.parent_account_id === parentUser.id) {
        res.status(400).json({ 
          success: false, 
          message: 'This account is already linked to your family' 
        });
      } else {
        res.status(400).json({ 
          success: false, 
          message: 'This account is already linked to another family' 
        });
      }
      return;
    }

    // Prevent self-linking
    if (childProfile.id === parentUser.id) {
      res.status(400).json({ 
        success: false, 
        message: 'You cannot link your own account as a child' 
      });
      return;
    }

    // Get child's age
    const { data: childAgeData } = await supabaseAdmin
      .from('users_with_age')
      .select('calculated_age, is_minor')
      .eq('id', childProfile.id)
      .single();

    // Only allow linking minors (or adults who consent)
    // For now, we'll allow any user to be linked but log it
    const childAge = childAgeData?.calculated_age;
    const isChildMinor = childAgeData?.is_minor || (childAge !== null && childAge < 18);

    // Link the accounts
    const { error: updateError } = await supabaseAdmin
      .from('users')
      .update({
        parent_account_id: parentUser.id,
        updated_at: new Date().toISOString()
      })
      .eq('id', childProfile.id);

    if (updateError) {
      console.error('Failed to link accounts:', updateError);
      res.status(500).json({ success: false, message: 'Failed to link accounts' });
      return;
    }

    // Set parent as family admin if not already
    await supabaseAdmin
      .from('users')
      .update({ is_family_admin: true })
      .eq('id', parentUser.id);

    // Log compliance event
    await supabaseAdmin
      .from('earnings_compliance_log')
      .insert({
        user_id: childProfile.id,
        event_type: 'family_link_created',
        details: {
          parent_id: parentUser.id,
          parent_email: parentProfile.email,
          child_age: childAge,
          is_minor: isChildMinor
        },
        ip_address: req.headers['x-forwarded-for']?.toString().split(',')[0] || null
      });

    res.status(200).json({
      success: true,
      message: isChildMinor 
        ? `${childProfile.display_name || childEmail} has been added to your family. As a minor, their earnings will be held until they turn 18 or you can claim them.`
        : `${childProfile.display_name || childEmail} has been added to your family.`,
      childId: childProfile.id,
      childName: childProfile.display_name || childEmail
    });

  } catch (error) {
    console.error('Family link error:', error);
    res.status(500).json({ success: false, message: 'Internal server error' });
  }
}


