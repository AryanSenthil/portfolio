# 🚀 Enable GitHub Pages - CRITICAL STEPS

Your code is pushed to GitHub, but the site is blank because GitHub Pages needs to be manually enabled.

## ⚠️ REQUIRED: Enable GitHub Pages (One-Time Setup)

Follow these exact steps:

### Step 1: Go to Repository Settings
1. Open your browser
2. Go to: **https://github.com/AryanSenthil/portfolio/settings/pages**
3. You should see the "GitHub Pages" settings page

### Step 2: Configure Source
Under **"Build and deployment"** section:

**Source:** Select **"GitHub Actions"** from the dropdown

(DO NOT select "Deploy from a branch" - that's the old method)

### Step 3: Save
- The setting should save automatically
- You should see a message about GitHub Actions

### Step 4: Trigger the Workflow
Go to: **https://github.com/AryanSenthil/portfolio/actions**

You should see:
- A workflow run called "Deploy to GitHub Pages"
- It might be running already, or you can manually trigger it

If you don't see it running:
1. Click "Deploy to GitHub Pages" workflow on the left
2. Click "Run workflow" button on the right
3. Click the green "Run workflow" button

### Step 5: Wait for Deployment (2-3 minutes)
- Watch the workflow complete (green checkmark)
- Once complete, your site will be at: **https://aryansenthil.github.io/portfolio/**

---

## 🔍 Troubleshooting

### If the site is still blank:

**Check 1: Verify GitHub Pages is enabled**
- Go to https://github.com/AryanSenthil/portfolio/settings/pages
- Source should be "GitHub Actions"
- You should see "Your site is live at https://aryansenthil.github.io/portfolio/"

**Check 2: Verify workflow ran successfully**
- Go to https://github.com/AryanSenthil/portfolio/actions
- Click on the latest "Deploy to GitHub Pages" run
- All steps should have green checkmarks
- If any step failed, check the error logs

**Check 3: Check permissions**
- Go to https://github.com/AryanSenthil/portfolio/settings/actions
- Under "Workflow permissions", ensure "Read and write permissions" is selected

**Check 4: Clear browser cache**
- Hard refresh: Ctrl+Shift+R (Windows/Linux) or Cmd+Shift+R (Mac)
- Or try in incognito/private browsing

---

## ✅ Success Indicators

Once it's working, you should see:

1. **In GitHub Pages settings:**
   - "Your site is live at https://aryansenthil.github.io/portfolio/"
   - Green checkmark indicating successful deployment

2. **In GitHub Actions:**
   - Latest workflow run shows all green checkmarks
   - Deploy step completed successfully

3. **At the URL:**
   - https://aryansenthil.github.io/portfolio/ shows your portfolio
   - All motion design features work smoothly
   - Navigation, glass effects, animations all present

---

## 📞 If You're Still Having Issues

The most common issue is that GitHub Pages simply isn't enabled yet. Make sure you:

1. ✅ Go to Settings → Pages
2. ✅ Set Source to "GitHub Actions"
3. ✅ Wait for the workflow to complete
4. ✅ Hard refresh your browser

Your portfolio is ready - it just needs GitHub Pages enabled!
