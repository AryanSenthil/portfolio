# 🚀 Deployment Guide - GitHub Pages

Your portfolio with Apple Glass aesthetic motion design is ready to deploy!

## 📍 Your GitHub Pages URL

Once deployed, your portfolio will be live at:

**🌐 https://aryansenthil.github.io/portfolio/**

---

## 🎯 Quick Deploy (Automatic)

I've set up GitHub Actions for automatic deployment. Follow these steps:

### Step 1: Enable GitHub Pages

1. Go to your repository: https://github.com/AryanSenthil/portfolio
2. Click **Settings** (top right)
3. In the left sidebar, click **Pages**
4. Under **Source**, select:
   - Source: **GitHub Actions**
5. Save (if prompted)

### Step 2: Push Your Changes

```bash
# Stage all changes
git add .

# Commit with a message
git commit -m "Add Apple Glass aesthetic motion design"

# Push to GitHub
git push origin main
```

### Step 3: Wait for Deployment

- Go to the **Actions** tab in your repository
- You'll see the "Deploy to GitHub Pages" workflow running
- Wait for it to complete (usually 2-3 minutes)
- Once the green checkmark appears, your site is live!

---

## 🔧 Manual Deploy (Alternative)

If you prefer to deploy manually:

```bash
# Build the project
npm run build

# The dist/ folder contains your built site
# You can manually upload this to any static hosting service
```

---

## ✅ Verify Deployment

After deployment completes:

1. Visit: **https://aryansenthil.github.io/portfolio/**
2. Test all the motion design features:
   - ✨ Glass morphism effects
   - ⚡ Smooth entrance animations
   - 🎯 Hover microinteractions
   - 📱 Responsive mobile menu
   - 🌙 Dark mode toggle

---

## 🔄 Update Your Site

To update your live site in the future:

```bash
# Make your changes, then:
git add .
git commit -m "Your update message"
git push origin main

# GitHub Actions will automatically rebuild and deploy
```

---

## 🐛 Troubleshooting

### Issue: Site not loading / 404 error

**Solution:**
1. Check GitHub Pages is enabled (Settings → Pages → Source: GitHub Actions)
2. Verify the workflow ran successfully (Actions tab)
3. Wait 5 minutes for DNS propagation

### Issue: Assets not loading / broken styles

**Solution:**
- The `base: '/portfolio/'` in `vite.config.js` handles this
- Already configured correctly ✅

### Issue: Workflow fails

**Solution:**
1. Check Actions tab for error details
2. Ensure `package-lock.json` is committed
3. Verify all dependencies install correctly locally

---

## 📊 What Gets Deployed

Your production build includes:

✅ All motion design enhancements
✅ Glass morphism effects
✅ Optimized JavaScript (108 KB gzipped)
✅ Optimized CSS (6.5 KB gzipped)
✅ Fast loading performance
✅ Full dark mode support
✅ Reduced motion accessibility
✅ Mobile responsive design

---

## 🎨 Motion Design Features Live

Once deployed, visitors will experience:

- 🪟 **Glass Morphism** - Translucent cards with backdrop blur
- ⚡ **Microinteractions** - Icon rotations, hover lifts, scale animations
- 🎯 **Timeline Animations** - Drawing effects, pulsing dots
- 📝 **Form Polish** - Focus glows, smooth transitions
- 🎪 **Staggered Entrances** - Sequential fade-ins
- ♿ **Accessibility** - Full reduced motion support
- 🚀 **60fps Animations** - GPU-accelerated smooth motion

---

## 💡 Tips

1. **Test locally first**: Always run `npm run dev` to verify changes before pushing
2. **Build check**: Run `npm run build` to catch any build errors locally
3. **Monitor Actions**: Check the Actions tab after pushing to see deployment status
4. **Share the link**: Once live, share https://aryansenthil.github.io/portfolio/ with professors and colleagues!

---

## 📞 Need Help?

If you encounter issues:
1. Check the Actions tab for workflow logs
2. Verify GitHub Pages settings
3. Test the build locally with `npm run build && npm run preview`

---

**Your portfolio is ready to impress academic professionals with its sophisticated Apple Glass aesthetic! 🎓✨**
