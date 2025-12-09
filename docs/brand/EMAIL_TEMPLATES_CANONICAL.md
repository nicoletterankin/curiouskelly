# ✨ Kelly Email Templates — Canonical

> All emails must embody Kelly's voice: Humble, Curious, Collaborative, Warm, Simple, Rich.

---

## Supabase Auth Templates

Copy these EXACTLY into Supabase → Authentication → Email Templates

---

### 1. Confirm Sign Up

**Subject:**
```
Let's learn something together
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi — I'm Kelly.<br><br>

I don't have all the answers. But I love finding them. And I think learning is better together.<br><br>

Every day I find something wonderful and I can't wait to share it.<br><br>

<a href="{{ .ConfirmationURL }}" style="color: #1e3a5f; text-decoration: underline;">Want to come along?</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 2. Magic Link

**Subject:**
```
Your door is open
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi — it's Kelly.<br><br>

I saved your spot. Today's lesson is ready whenever you are.<br><br>

<a href="{{ .ConfirmationURL }}" style="color: #1e3a5f; text-decoration: underline;">Come on in.</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 3. Reset Password

**Subject:**
```
Let's get you back in
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi — it's Kelly.<br><br>

Happens to everyone. Let's get you back to learning.<br><br>

<a href="{{ .ConfirmationURL }}" style="color: #1e3a5f; text-decoration: underline;">Reset your password</a><br><br>

I'll be here when you're ready.<br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 4. Invite User

**Subject:**
```
Someone thought of you
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi — I'm Kelly.<br><br>

Someone who cares about you thought you might like learning together with us.<br><br>

Every day I find something wonderful — a small truth about how the world works — and I share it. Five minutes. That's all it takes.<br><br>

<a href="{{ .ConfirmationURL }}" style="color: #1e3a5f; text-decoration: underline;">Want to see what it's about?</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 5. Change Email Address

**Subject:**
```
Quick confirmation
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi — it's Kelly.<br><br>

Just making sure this is really you. One click and we're all set.<br><br>

<a href="{{ .ConfirmationURL }}" style="color: #1e3a5f; text-decoration: underline;">Confirm your new email</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 6. Reauthentication

**Subject:**
```
Just checking it's you
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi — it's Kelly.<br><br>

Before we continue, I just want to make sure it's really you.<br><br>

Your code is: <strong style="font-size: 24px; letter-spacing: 2px;">{{ .Token }}</strong><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

## Application Emails (via Resend API)

---

### 7. Welcome Email (After Signup)

**Subject:**
```
Let's learn something together
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi — I'm Kelly.<br><br>

I don't have all the answers. But I love finding them. And I think learning is better together.<br><br>

Every day I find something wonderful and I can't wait to share it. Today's lesson is ready.<br><br>

<a href="https://curiouskelly.com/learn" style="color: #1e3a5f; text-decoration: underline;">Want to come along?</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 8. Daily Lesson Email

**Subject:**
```
Today's lesson is ready
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Good morning.<br><br>

I found something wonderful today: <strong>{{ .LessonTitle }}</strong><br><br>

Five minutes. I think you'll love it.<br><br>

<a href="{{ .LessonURL }}" style="color: #1e3a5f; text-decoration: underline;">Let's learn together.</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 9. Streak Celebration (7 days)

**Subject:**
```
One week together
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi {{ .Name }} —<br><br>

Seven days. That's how long we've been learning together now.<br><br>

I just wanted to say: I notice. And it matters.<br><br>

Most people give up. You didn't. That's rare. That's beautiful.<br><br>

See you tomorrow?<br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 10. Streak Celebration (30 days)

**Subject:**
```
A whole month
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

{{ .Name }} —<br><br>

30 days.<br><br>

30 times you showed up. 30 lessons. 30 moments of choosing curiosity over everything else competing for your attention.<br><br>

I don't take that for granted. Thank you for learning with me.<br><br>

Here's to the next 30.<br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 11. Miss You (Re-engagement)

**Subject:**
```
Your spot is still here
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi {{ .Name }} —<br><br>

It's been a little while. No guilt — life happens. I just wanted you to know your spot is still here.<br><br>

Whenever you're ready, I'll be ready too.<br><br>

<a href="https://curiouskelly.com/learn" style="color: #1e3a5f; text-decoration: underline;">Come back anytime.</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 12. Birthday

**Subject:**
```
Happy birthday, {{ .Name }}
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

{{ .Name }} —<br><br>

Today is yours.<br><br>

I hope it's filled with people who love you, moments that surprise you, and at least one thing that makes you wonderfully curious.<br><br>

Happy birthday from someone who's grateful you exist.<br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 13. Affiliate Welcome

**Subject:**
```
Welcome, partner
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi {{ .Name }} —<br><br>

Thank you for believing in what we're building.<br><br>

I started this because I think curiosity makes people kinder, sharper, more alive. And now you're helping spread that. That means something.<br><br>

Your referral code is: <strong>{{ .AffiliateCode }}</strong><br><br>

Share it with people you think would love learning. You'll earn 30% of every subscription, for as long as they stay curious.<br><br>

<a href="https://curiouskelly.com/affiliate/dashboard" style="color: #1e3a5f; text-decoration: underline;">See your dashboard</a><br><br>

Let's do something good together.<br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

### 14. Affiliate Payout

**Subject:**
```
Your earnings are on the way
```

**Body:**
```html
<p style="font-family: Georgia, serif; font-size: 19px; color: #1f2937; line-height: 1.9; max-width: 460px;">

Hi {{ .Name }} —<br><br>

Good news: <strong>{{ .Amount }}</strong> is on its way to you.<br><br>

That's from {{ .ReferralCount }} people who are now learning every day because of you. Thank you for spreading curiosity.<br><br>

<a href="https://curiouskelly.com/affiliate/payouts" style="color: #1e3a5f; text-decoration: underline;">View details</a><br><br>

<span style="color: #6b7280;">— Kelly</span>

</p>
```

---

## Voice Checklist

Before sending ANY email, verify:

- [ ] Sounds like Kelly, not a brand
- [ ] Humble, not superior
- [ ] Inviting, not demanding
- [ ] Simple, not clever
- [ ] Rich, not cheap
- [ ] Says "learner" not "user"
- [ ] Focused on learning TOGETHER

---

*These templates are CANONICAL. Do not modify without voice review.*


