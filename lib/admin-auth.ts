/**
 * Admin Authentication Helper
 * 
 * Verifies that a request is from an authenticated admin user.
 * Used to protect CFO dashboard, video approval, and other admin endpoints.
 */

import type { VercelRequest } from '@vercel/node';
import { createClient } from '@supabase/supabase-js';

const supabaseUrl = process.env.PUBLIC_SUPABASE_URL || process.env.NEXT_PUBLIC_SUPABASE_URL;
const supabaseServiceKey = process.env.SUPABASE_SERVICE_ROLE_KEY;

// Admin user IDs from environment (comma-separated)
const ADMIN_USER_IDS = (process.env.ADMIN_USER_IDS || '').split(',').filter(Boolean);

export interface AdminAuthResult {
  authenticated: boolean;
  isAdmin: boolean;
  userId?: string;
  email?: string;
  error?: string;
}

/**
 * Check if a request is from an authenticated admin
 * 
 * Returns detailed auth result for logging/debugging.
 * Use `requireAdmin()` for simple boolean check.
 */
export async function checkAdminAuth(req: VercelRequest): Promise<AdminAuthResult> {
  const authHeader = req.headers.authorization;
  
  if (!authHeader || !authHeader.startsWith('Bearer ')) {
    return {
      authenticated: false,
      isAdmin: false,
      error: 'Missing or invalid Authorization header',
    };
  }
  
  if (!supabaseUrl || !supabaseServiceKey) {
    return {
      authenticated: false,
      isAdmin: false,
      error: 'Supabase not configured',
    };
  }
  
  const token = authHeader.replace('Bearer ', '');
  const supabase = createClient(supabaseUrl, supabaseServiceKey);
  
  try {
    // Verify token and get user
    const { data: { user }, error: authError } = await supabase.auth.getUser(token);
    
    if (authError || !user) {
      return {
        authenticated: false,
        isAdmin: false,
        error: 'Invalid or expired token',
      };
    }
    
    // Check if user is admin via:
    // 1. Environment variable whitelist
    // 2. Database is_admin flag
    
    let isAdmin = ADMIN_USER_IDS.includes(user.id);
    
    if (!isAdmin) {
      // Check database
      const { data: profile, error: profileError } = await supabase
        .from('users')
        .select('is_admin')
        .eq('id', user.id)
        .single();
      
      if (!profileError && profile?.is_admin === true) {
        isAdmin = true;
      }
    }
    
    return {
      authenticated: true,
      isAdmin,
      userId: user.id,
      email: user.email,
      error: isAdmin ? undefined : 'User is not an admin',
    };
    
  } catch (error) {
    return {
      authenticated: false,
      isAdmin: false,
      error: error instanceof Error ? error.message : 'Authentication failed',
    };
  }
}

/**
 * Simple boolean check - is this request from an admin?
 * 
 * Usage:
 * ```ts
 * if (!await requireAdmin(req)) {
 *   return res.status(401).json({ error: 'Admin access required' });
 * }
 * ```
 */
export async function requireAdmin(req: VercelRequest): Promise<boolean> {
  const result = await checkAdminAuth(req);
  return result.authenticated && result.isAdmin;
}

/**
 * Express/Vercel middleware for admin routes
 * 
 * Usage in API route:
 * ```ts
 * const authResult = await adminMiddleware(req, res);
 * if (!authResult) return; // Response already sent
 * // ... proceed with admin-only logic
 * ```
 */
export async function adminMiddleware(
  req: VercelRequest,
  res: any
): Promise<AdminAuthResult | null> {
  const result = await checkAdminAuth(req);
  
  if (!result.authenticated) {
    res.status(401).json({
      error: 'unauthorized',
      message: 'Authentication required',
      details: result.error,
    });
    return null;
  }
  
  if (!result.isAdmin) {
    res.status(403).json({
      error: 'forbidden',
      message: 'Admin access required',
    });
    return null;
  }
  
  return result;
}

/**
 * Get user info from request (authenticated but not necessarily admin)
 */
export async function getUserFromRequest(req: VercelRequest): Promise<{
  userId: string;
  email: string;
} | null> {
  const authHeader = req.headers.authorization;
  
  if (!authHeader?.startsWith('Bearer ') || !supabaseUrl || !supabaseServiceKey) {
    return null;
  }
  
  const token = authHeader.replace('Bearer ', '');
  const supabase = createClient(supabaseUrl, supabaseServiceKey);
  
  try {
    const { data: { user }, error } = await supabase.auth.getUser(token);
    if (error || !user) return null;
    
    return {
      userId: user.id,
      email: user.email || '',
    };
  } catch {
    return null;
  }
}
