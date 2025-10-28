// Configuration file for API endpoints
// Update this URL to match your backend server location

const CONFIG = {
    // API Configuration - Use relative URLs for same-origin deployment
    API_BASE_URL: window.location.origin,
    
    // API Endpoints
    API_ENDPOINTS: {
        CLASSIFY: 'classify',
        NEXTGEN: 'nextgen'
    },
    
    // Audio folder paths
    AUDIO_FOLDERS: {
        TABLA: 'tabla',
        DRUMS: 'drums'
    }
};

// Export for use in other files
window.CONFIG = CONFIG;
