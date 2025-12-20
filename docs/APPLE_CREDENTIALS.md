# Apple Sign-In Credentials

## ⚠️ SENSITIVE - Do Not Share Publicly

These credentials are required to configure Apple Sign-In in Supabase.

## Credentials Summary

| Field | Value |
|-------|-------|
| **Services ID** | `com.curiouskelly.signin` |
| **Team ID** | `V4K3TZM9QP` |
| **Key ID** | `Y95GAVCDQ2` |
| **Key File** | `AuthKey_Y95GAVCDQ2.p8` |

## How to Configure in Supabase

1. Go to: https://supabase.com/dashboard/project/tvjalxxsyryjphkforjv/auth/providers
2. Scroll to **"Apple"** provider
3. Toggle **"Enable Apple provider"** to ON
4. Fill in the fields:

### Apple Client ID (Services ID)
```
com.curiouskelly.signin
```

### Apple Secret (Team ID)
```
V4K3TZM9QP
```

### Apple Key ID
```
Y95GAVCDQ2
```

### Apple Private Key
1. Open the file `AuthKey_Y95GAVCDQ2.p8` in a text editor (Notepad)
2. Copy **ALL** the contents, including the BEGIN and END lines
3. Paste into this field

The content should look like:
```
-----BEGIN PRIVATE KEY-----
[Multiple lines of base64 encoded text]
-----END PRIVATE KEY-----
```

## Configuration Checklist

- [ ] Opened `AuthKey_Y95GAVCDQ2.p8` in text editor
- [ ] Copied entire key content
- [ ] Navigated to Supabase Auth Providers
- [ ] Enabled Apple provider
- [ ] Entered Services ID: `com.curiouskelly.signin`
- [ ] Entered Team ID: `V4K3TZM9QP`
- [ ] Entered Key ID: `Y95GAVCDQ2`
- [ ] Pasted private key content
- [ ] Clicked Save
- [ ] Tested Apple Sign-In from `index.html`

## Already Configured in Apple Developer

✅ Return URL: `https://tvjalxxsyryjphkforjv.supabase.co/auth/v1/callback`
✅ Domain: `tvjalxxsyryjphkforjv.supabase.co`
✅ Primary App ID: `V4K3TZM9QP.ilearn.love`
✅ Services ID: `com.curiouskelly.signin`

## Troubleshooting

If Apple Sign-In doesn't work:
1. Verify all credentials are entered correctly in Supabase
2. Check that the private key includes BEGIN/END lines
3. Ensure the redirect URL in `public/index.html` matches: `/public/app.html`
4. Check browser console for any OAuth errors

## Security Notes

- The `.p8` key file can only be downloaded once from Apple
- Keep the key file secure and backed up
- Never commit the private key to git
- This file (`APPLE_CREDENTIALS.md`) should not be committed to public repos



































