# Rudra's Personal Website 🚀

[![Build & Deploy](https://github.com/rgs2151/landing/actions/workflows/jekyll.yml/badge.svg)](https://github.com/rgs2151/landing/actions/workflows/jekyll.yml)

A minimal, modern portfolio website featuring a two-column layout with bio and works showcase.

## 🎨 Features

- **Two-column layout**: Bio on the left, works/projects on the right
- **External links**: All posts and projects link to external platforms (LessWrong, arXiv, etc.)
- **Card-based design**: Clean cards for each work with image, title, description, and metadata
- **Dark theme**: Columbia Blue accent color (#B9D9EB)
- **Responsive**: Mobile-friendly with automatic single-column layout on smaller screens
- **Fast**: Static site with no bloat

## 🚀 Quick Start

### Development

```bash
# Start development server with live reload
bundle exec jekyll serve --livereload

# Build for production
bundle exec jekyll build
```

### Adding New Works

1. **Add an image** to `/assets/images/[work-name]/preview.jpg`
2. **Edit** `_data/works.yml` and add your entry:

```yaml
- title: "Your Work Title"
  description: "Brief description (2-3 sentences)"
  image: "/assets/images/work-name/preview.jpg"
  url: "https://external-link.com"
  type: "post"  # Options: post, paper, project
  date: "2025-XX-XX"
```

3. **Commit and push** - GitHub Actions will automatically deploy

## 📁 Project Structure

```
├── _config.yml           # Site configuration
├── _data/
│   └── works.yml        # All your works/projects
├── _includes/           # Reusable components
├── _layouts/
│   └── default.html     # Main layout template
├── _sass/
│   ├── _content.scss    # Main styles
│   └── _vs.scss         # Code highlighting
├── assets/
│   ├── css/
│   └── images/          # Work preview images
└── index.md             # Homepage bio
```

## �� Customization

### Update Your Bio
Edit `index.md` to change your bio text

### Change Colors
Edit CSS variables in `_sass/_content.scss`:
```scss
:root {
    --accent: #B9D9EB;  /* Your accent color */
    --bg: hsl(0, 0%, 0%);  /* Background color */
    // ... more variables
}
```

### Update Links
Edit the links in `_layouts/default.html` under `.bio__links`

## 📝 Notes

- No blog functionality - all content links externally
- No navigation bar - simple one-page portfolio
- Images should be ~1200x630px for best results
- Site rebuilds automatically on push to main branch


