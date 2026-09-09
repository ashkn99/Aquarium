# Deploying the beta

Free hosting, no credit card required for either service. Takes about
10 minutes.

## 1. Push this repo to GitHub

If it isn't on GitHub yet:

```bash
git remote add origin https://github.com/<your-username>/aquarium-assistant.git
git push -u origin master
```

(Create the empty repo on github.com first — pick public or private,
either works with Render's free tier.)

## 2. Create a free Postgres database (Neon)

1. Go to https://neon.tech, sign up (GitHub login is fine), create a project.
2. On the project dashboard, copy the **connection string** shown there
   (starts with `postgresql://`). Keep this tab open, you'll paste it in
   step 3.

Render's own free tier disk is wiped on every redeploy/restart, so this
is where all case/observation/feedback data actually lives long-term.

## 3. Deploy to Render

1. Go to https://render.com, sign up (GitHub login is fine).
2. **New > Blueprint**, connect your GitHub account, pick this repo.
   Render reads `render.yaml` at the repo root and pre-fills the build
   command, start command, and plan (free) automatically.
3. When prompted for the `DATABASE_URL` environment variable, paste the
   Neon connection string from step 2.
4. Click deploy. First build takes a few minutes (installing
   dependencies). Render gives you a `https://<something>.onrender.com`
   URL when it's live.

Free-tier note: the service spins down after ~15 minutes of no traffic
and takes ~30-50 seconds to wake back up on the next request. Fine for a
beta; upgradeable later if traffic justifies it.

## 4. Verify

1. Open the `*.onrender.com` URL, click through one full case, submit
   feedback at the end.
2. In the Neon dashboard's SQL editor, run
   `select * from cases order by created_at desc limit 5;` and confirm
   your case shows up.
3. From your machine (with `DATABASE_URL` set to the same Neon connection
   string), run:
   ```bash
   python -m aqua_assistant.analytics.report
   ```
   and confirm it picks up that case.

## 5. Share the link

The `*.onrender.com` URL is the public beta link — no login required,
per the "public open link" access model this was built for.

## Later, once validated

- Custom domain: Render supports adding one for free once you own it
  (paid, per your own domain registrar) — TLS is automatic either way.
- Removing the cold-start delay: upgrade the Render service off the free
  plan.
