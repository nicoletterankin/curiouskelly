import { neon, NeonQueryFunction } from '@neondatabase/serverless'
import { cookies } from 'next/headers'
import { SignJWT, jwtVerify } from 'jose'

// Lazy-initialized SQL client - prevents build-time connection errors
let _sql: NeonQueryFunction<false, false> | null = null
function getSql() {
  if (!_sql) {
    _sql = neon(process.env.DATABASE_URL!)
  }
  return _sql
}

// JWT secret - MUST be set via environment variable
const JWT_SECRET = new TextEncoder().encode(
  process.env.JWT_SECRET
)

// Validate at module level that JWT_SECRET is configured
if (!process.env.JWT_SECRET && process.env.NODE_ENV === 'production') {
  console.error('[auth] CRITICAL: JWT_SECRET environment variable is not set. Authentication will fail.')
}

const SESSION_COOKIE = 'kelly_session'
const SESSION_EXPIRY = 60 * 60 * 24 * 30 // 30 days in seconds

export interface User {
  id: string
  email: string
  name: string | null
  age: number | null
  language: string
  archetype: string
  dayOfYear: number
  subscriptionStatus: string
  createdAt: Date
}

export interface SessionPayload {
  userId: string
  email: string
  exp: number
}

// Password hashing using Web Crypto PBKDF2 (iterative, resistant to brute-force)
// Uses 100,000 iterations with a per-password random salt
async function hashPassword(password: string): Promise<string> {
  const salt = crypto.getRandomValues(new Uint8Array(16))
  const encoder = new TextEncoder()
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    encoder.encode(password),
    { name: 'PBKDF2' },
    false,
    ['deriveBits']
  )
  const hashBuffer = await crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      salt,
      iterations: 100_000,
      hash: 'SHA-256',
    },
    keyMaterial,
    256
  )
  const saltHex = Array.from(salt).map(b => b.toString(16).padStart(2, '0')).join('')
  const hashHex = Array.from(new Uint8Array(hashBuffer)).map(b => b.toString(16).padStart(2, '0')).join('')
  // Store as salt:hash so we can verify later
  return `${saltHex}:${hashHex}`
}

async function verifyPassword(password: string, storedHash: string): Promise<boolean> {
  // Support legacy SHA-256 hashes (no colon separator) during migration
  if (!storedHash.includes(':')) {
    const encoder = new TextEncoder()
    const data = encoder.encode(password + (process.env.PASSWORD_SALT || 'kelly-salt'))
    const hashBuffer = await crypto.subtle.digest('SHA-256', data)
    const hashArray = Array.from(new Uint8Array(hashBuffer))
    const legacyHash = hashArray.map(b => b.toString(16).padStart(2, '0')).join('')
    return legacyHash === storedHash
  }

  const [saltHex, expectedHash] = storedHash.split(':')
  const salt = new Uint8Array(saltHex.match(/.{2}/g)!.map(byte => Number.parseInt(byte, 16)))
  const encoder = new TextEncoder()
  const keyMaterial = await crypto.subtle.importKey(
    'raw',
    encoder.encode(password),
    { name: 'PBKDF2' },
    false,
    ['deriveBits']
  )
  const hashBuffer = await crypto.subtle.deriveBits(
    {
      name: 'PBKDF2',
      salt,
      iterations: 100_000,
      hash: 'SHA-256',
    },
    keyMaterial,
    256
  )
  const hashHex = Array.from(new Uint8Array(hashBuffer)).map(b => b.toString(16).padStart(2, '0')).join('')
  return hashHex === expectedHash
}

// Create JWT session token
async function createSessionToken(userId: string, email: string): Promise<string> {
  return new SignJWT({ userId, email })
    .setProtectedHeader({ alg: 'HS256' })
    .setIssuedAt()
    .setExpirationTime(`${SESSION_EXPIRY}s`)
    .sign(JWT_SECRET)
}

// Verify JWT session token
async function verifySessionToken(token: string): Promise<SessionPayload | null> {
  try {
    const { payload } = await jwtVerify(token, JWT_SECRET)
    return payload as unknown as SessionPayload
  } catch {
    return null
  }
}

// Set session cookie
export async function setSessionCookie(userId: string, email: string): Promise<void> {
  const token = await createSessionToken(userId, email)
  const cookieStore = await cookies()
  cookieStore.set(SESSION_COOKIE, token, {
    httpOnly: true,
    secure: process.env.NODE_ENV === 'production',
    sameSite: 'lax',
    maxAge: SESSION_EXPIRY,
    path: '/',
  })
}

// Clear session cookie
export async function clearSessionCookie(): Promise<void> {
  const cookieStore = await cookies()
  cookieStore.delete(SESSION_COOKIE)
}

// Get current session
export async function getSession(): Promise<SessionPayload | null> {
  const cookieStore = await cookies()
  const token = cookieStore.get(SESSION_COOKIE)?.value
  if (!token) return null
  return verifySessionToken(token)
}

// Get current user from session
export async function getCurrentUser(): Promise<User | null> {
  const session = await getSession()
  if (!session) return null
  
  try {
    const sql = getSql()
    const users = await sql`
      SELECT id, email, name, age, language, archetype, day_of_year, 
             subscription_status, created_at
      FROM learners
      WHERE id = ${session.userId}
      LIMIT 1
    `
    
    if (users.length === 0) return null
    
    const u = users[0]
    return {
      id: u.id as string,
      email: u.email as string,
      name: u.name as string | null,
      age: u.age as number | null,
      language: (u.language as string) || 'en',
      archetype: (u.archetype as string) || 'storyteller',
      dayOfYear: (u.day_of_year as number) || 1,
      subscriptionStatus: (u.subscription_status as string) || 'trial',
      createdAt: new Date(u.created_at as string),
    }
  } catch (error) {
    console.error('[auth] Error fetching user:', error)
    return null
  }
}

// Register new user
export async function registerUser(
  email: string, 
  password: string, 
  name?: string
): Promise<{ success: boolean; error?: string; user?: User }> {
  try {
    const sql = getSql()
    // Check if user exists
    const existing = await sql`
      SELECT id FROM learners WHERE email = ${email.toLowerCase()}
    `
    
    if (existing.length > 0) {
      return { success: false, error: 'Email already registered' }
    }
    
    // Hash password
    const passwordHash = await hashPassword(password)
    
    // Create user
    const result = await sql`
      INSERT INTO learners (
        email, password_hash, name, language, archetype, day_of_year,
        subscription_status, created_at
      ) VALUES (
        ${email.toLowerCase()}, ${passwordHash}, ${name || null}, 
        'en', 'storyteller', 1, 'trial', NOW()
      )
      RETURNING id, email, name, age, language, archetype, day_of_year, 
                subscription_status, created_at
    `
    
    if (result.length === 0) {
      return { success: false, error: 'Failed to create user' }
    }
    
    const u = result[0]
    const user: User = {
      id: u.id as string,
      email: u.email as string,
      name: u.name as string | null,
      age: u.age as number | null,
      language: (u.language as string) || 'en',
      archetype: (u.archetype as string) || 'storyteller',
      dayOfYear: (u.day_of_year as number) || 1,
      subscriptionStatus: (u.subscription_status as string) || 'trial',
      createdAt: new Date(u.created_at as string),
    }
    
    // Set session cookie
    await setSessionCookie(user.id, user.email)
    
    return { success: true, user }
  } catch (error) {
    console.error('[auth] Registration error:', error)
    return { success: false, error: 'Registration failed' }
  }
}

// Login user
export async function loginUser(
  email: string, 
  password: string
): Promise<{ success: boolean; error?: string; user?: User }> {
  try {
    const sql = getSql()
    const users = await sql`
      SELECT id, email, name, age, language, archetype, day_of_year,
             subscription_status, password_hash, created_at
      FROM learners
      WHERE email = ${email.toLowerCase()}
      LIMIT 1
    `
    
    if (users.length === 0) {
      return { success: false, error: 'Invalid email or password' }
    }
    
    const u = users[0]
    const isValid = await verifyPassword(password, u.password_hash as string)
    
    if (!isValid) {
      return { success: false, error: 'Invalid email or password' }
    }
    
    const user: User = {
      id: u.id as string,
      email: u.email as string,
      name: u.name as string | null,
      age: u.age as number | null,
      language: (u.language as string) || 'en',
      archetype: (u.archetype as string) || 'storyteller',
      dayOfYear: (u.day_of_year as number) || 1,
      subscriptionStatus: (u.subscription_status as string) || 'trial',
      createdAt: new Date(u.created_at as string),
    }
    
    // Set session cookie
    await setSessionCookie(user.id, user.email)
    
    return { success: true, user }
  } catch (error) {
    console.error('[auth] Login error:', error)
    return { success: false, error: 'Login failed' }
  }
}

// Logout user
export async function logoutUser(): Promise<void> {
  await clearSessionCookie()
}

// Update user profile
export async function updateUserProfile(
  userId: string,
  updates: Partial<Pick<User, 'name' | 'age' | 'language' | 'archetype'>>
): Promise<{ success: boolean; error?: string }> {
  try {
    const sql = getSql()
    await sql`
      UPDATE learners
      SET 
        name = COALESCE(${updates.name}, name),
        age = COALESCE(${updates.age}, age),
        language = COALESCE(${updates.language}, language),
        archetype = COALESCE(${updates.archetype}, archetype),
        updated_at = NOW()
      WHERE id = ${userId}
    `
    return { success: true }
  } catch (error) {
    console.error('[auth] Update profile error:', error)
    return { success: false, error: 'Failed to update profile' }
  }
}
