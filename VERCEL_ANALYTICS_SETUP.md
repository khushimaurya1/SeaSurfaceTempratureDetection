# Vercel Web Analytics Setup Guide

This document explains how Vercel Web Analytics has been integrated into the Sea Surface Temperature Detection application.

## What Was Implemented

### 1. HTML Template Created
- **File**: `templates/index.html`
- A complete, responsive web interface for the Flask application
- Includes all functionality: data generation, temperature analysis, event detection, and thermal image analysis
- Modern UI with gradient backgrounds, interactive controls, and visualization displays

### 2. Vercel Analytics Integration
The following code was added to `templates/index.html`:

```html
<!-- Vercel Web Analytics -->
<script>
    window.va = window.va || function () { (window.vaq = window.vaq || []).push(arguments); };
</script>
<script defer src="/_vercel/insights/script.js"></script>
```

This integration follows Vercel's official documentation for HTML/static sites.

### 3. Vercel Deployment Configuration
- **File**: `vercel.json`
- Configures Vercel to deploy the Flask Python application
- Routes all requests to `app.py`

## How It Works

### Analytics Script
When deployed to Vercel:
1. The `/_vercel/insights/script.js` path is automatically provided by Vercel's platform
2. The script tracks page views and web vitals automatically
3. No additional configuration is required in the code

### Page Tracking
The analytics will automatically track:
- **Page Views**: When users visit the application
- **Web Vitals**: Performance metrics (LCP, FID, CLS, etc.)
- **Custom Events**: Can be added using `window.va('event', { name: 'event_name' })`

## Enabling Analytics in Vercel Dashboard

To start collecting analytics data:

1. **Deploy to Vercel**:
   ```bash
   vercel deploy
   ```

2. **Enable Web Analytics**:
   - Go to your project in the Vercel Dashboard
   - Navigate to the "Analytics" tab
   - Click "Enable Web Analytics"
   - Analytics will start collecting data after the next deployment

3. **View Analytics**:
   - After deployment and some traffic, visit the Analytics tab
   - View metrics for page views, visitors, and web vitals
   - Data typically appears within a few hours

## Testing the Integration

### Local Testing
When running locally (`python app.py` or `python app_simple.py`), the analytics script won't load because the `/_vercel/insights/script.js` path is only available on Vercel's platform. This is expected behavior and won't cause any errors.

### Production Testing
After deploying to Vercel:
1. Open your browser's Developer Tools (F12)
2. Go to the Network tab
3. Visit your deployed application
4. Look for a request to `/_vercel/insights/script.js` - it should load successfully (200 status)
5. You may also see POST requests to `/_vercel/insights/event` for tracked events

## Project Structure

```
.
├── app.py                          # Main Flask application (full version)
├── app_simple.py                   # Simplified Flask application
├── seasurface.py                   # SST detection logic
├── requirements.txt                # Python dependencies
├── vercel.json                     # Vercel deployment configuration
├── templates/
│   └── index.html                  # Web UI with Vercel Analytics
├── static/                         # Static assets directory (empty)
└── VERCEL_ANALYTICS_SETUP.md      # This file
```

## Features of the Web Interface

The HTML template includes:

1. **Data Generation**: Generate synthetic SST data for a specified number of days
2. **Temperature Analysis**: Visualize temperature maps, time series, and anomalies
3. **Event Detection**: Identify marine heatwave events with configurable thresholds
4. **Thermal Image Analysis**: Analyze thermal images for hotspots and cold spots
5. **Responsive Design**: Works on desktop and mobile devices
6. **Loading States**: Visual feedback during API calls
7. **Error Handling**: User-friendly error messages

## Advanced Analytics Configuration (Optional)

You can add custom event tracking by using the `window.va` function:

```javascript
// Track custom events
window.va('event', { 
    name: 'data_generated',
    data: { days: 30 }
});

// Filter events before sending (privacy)
window.va('beforeSend', (event) => {
    // Return null to prevent sending
    if (event.url.includes('/private')) {
        return null;
    }
    return event;
});
```

## Troubleshooting

### Analytics not showing data
- Ensure you've enabled Web Analytics in the Vercel Dashboard
- Wait a few hours for data to populate
- Check that your deployment was successful
- Verify the script loads in the browser's Network tab

### Script not loading
- Confirm you're testing on a Vercel deployment (not localhost)
- Check that `vercel.json` is properly configured
- Ensure your project is linked to a Vercel account

### Console errors
- The `window.va` function is defined before the script loads, so no errors should occur
- If you see errors, check the browser console for details

## Documentation References

- [Vercel Analytics Quickstart](https://vercel.com/docs/analytics/quickstart)
- [Vercel Analytics Package Documentation](https://vercel.com/docs/analytics/package)
- [Flask Documentation](https://flask.palletsprojects.com/)

## Support

For Vercel Analytics issues:
- Visit [Vercel Documentation](https://vercel.com/docs)
- Check [Vercel Community](https://github.com/vercel/vercel/discussions)

For application-specific issues:
- Check the Flask application logs
- Review the browser console for JavaScript errors
- Verify API endpoints are responding correctly
