// File upload application logic
let selectedFiles = null;

// Get DOM elements
const fileInput = document.getElementById('fileInput');
const fileInfo = document.getElementById('fileInfo');
const fileName = document.getElementById('fileName');
const fileSize = document.getElementById('fileSize');
const submitButtons = document.querySelectorAll('.submit-btn');
const statusMessage = document.getElementById('statusMessage');

// Handle file selection
fileInput.addEventListener('change', function(event) {
    const files = event.target.files;
    
    if (files.length > 0) {
        selectedFiles = files;
        displayFileInfo(files);
        submitButtons.forEach(btn => btn.disabled = false);
        hideStatusMessage();
    } else {
        selectedFiles = null;
        hideFileInfo();
        submitButtons.forEach(btn => btn.disabled = true);
    }
});

// Display file information
function displayFileInfo(files) {
    const file = files[0];
    fileName.textContent = `File: ${file.name}`;
    fileSize.textContent = `Size: ${formatFileSize(file.size)}`;
    fileInfo.style.display = 'block';
}

// Hide file information
function hideFileInfo() {
    fileInfo.style.display = 'none';
}

// Format file size
function formatFileSize(bytes) {
    if (bytes === 0) return '0 Bytes';
    
    const k = 1024;
    const sizes = ['Bytes', 'KB', 'MB', 'GB'];
    const i = Math.floor(Math.log(bytes) / Math.log(k));
    
    return parseFloat((bytes / Math.pow(k, i)).toFixed(2)) + ' ' + sizes[i];
}

// Handle submit button clicks
submitButtons.forEach(button => {
    button.addEventListener('click', function() {
        if (!selectedFiles || selectedFiles.length === 0) {
            showStatusMessage('No files selected!', 'error');
            return;
        }
        
        const apiEndpoint = this.dataset.api;
        const buttonText = this.textContent;
        
        // Show processing message
        showStatusMessage(`Processing with ${buttonText}...`, 'success');
        
        // Disable all buttons during processing
        submitButtons.forEach(btn => btn.disabled = true);
        
        // Process files with the selected API
        setTimeout(() => {
            processFiles(selectedFiles, apiEndpoint, buttonText, this.id);
        }, 1000);
    });
});

// Process the selected files
async function processFiles(files, apiEndpoint, buttonText, buttonId) {
    try {
        // We'll process the first file (you can modify this to handle multiple files)
        const file = files[0];
        
        // Create FormData object
        const formData = new FormData();
        formData.append('file', file);
        
        // Determine the API URL based on endpoint
        const apiUrl = `${CONFIG.API_BASE_URL}/${apiEndpoint}`;
        
        // Make API call
        console.log(`Uploading file to ${apiEndpoint} API:`, file.name);
        console.log(`Using endpoint: ${apiUrl}`);
        
        // Make API call
        const response = await fetch(apiUrl, {
            method: 'POST',
            body: formData
        });
        
        if (!response.ok) {
            throw new Error(`API request failed: ${response.status} ${response.statusText}`);
        }
        
        const result = await response.json();
        
        console.log('API Response:', result);
        
        // Handle different API responses
        if ((apiEndpoint === 'classify' || apiEndpoint === 'nextgen') && result.duration && result.notes) {
            showStatusMessage(
                `${buttonText} completed! Playing notes...`, 
                'success'
            );
            
            // Determine which folder to use based on button ID
            let soundFolder = CONFIG.AUDIO_FOLDERS.DRUMS; // default
            
            if (buttonId === 'classifyTablaBtn' || buttonId === 'nextgenTablaBtn') {
                soundFolder = CONFIG.AUDIO_FOLDERS.TABLA;
            } else if (buttonId === 'convertDrumsBtn' || buttonId === 'generateMusicBtn') {
                soundFolder = CONFIG.AUDIO_FOLDERS.DRUMS;
            }
            
            // Play the notes with their durations from the appropriate folder
            await playNotesSequence(result.duration, result.notes, soundFolder);
            
            showStatusMessage(
                `${buttonText} completed successfully! Notes played. You can perform other operations on the same file.`, 
                'success'
            );
            
            // Re-enable all submit buttons for additional operations
            setTimeout(() => {
                submitButtons.forEach(btn => btn.disabled = false);
            }, 500);
        } else {
            showStatusMessage(
                `${buttonText} completed successfully! Check console for results. You can perform other operations on the same file.`, 
                'success'
            );
            
            // Re-enable all submit buttons for additional operations
            setTimeout(() => {
                submitButtons.forEach(btn => btn.disabled = false);
            }, 500);
        }
        
        // Keep the file for additional operations
        // File will only be cleared when a new file is uploaded
        
    } catch (error) {
        console.error('Error processing files:', error);
        
        let errorMessage = 'Error processing files. Please try again.';
        if (error.name === 'TypeError' && error.message.includes('fetch')) {
            errorMessage = 'Cannot connect to API. Please check if the server is running.';
        } else if (error.message.includes('API request failed')) {
            errorMessage = `API Error: ${error.message}`;
        }
        
        showStatusMessage(errorMessage, 'error');
        
        // Re-enable buttons on error
        submitButtons.forEach(btn => btn.disabled = false);
    }
}

// Show status message
function showStatusMessage(message, type) {
    statusMessage.textContent = message;
    statusMessage.className = `status-message ${type}`;
    statusMessage.style.display = 'block';
}

// Hide status message
function hideStatusMessage() {
    statusMessage.style.display = 'none';
}

// Reset the form
function resetForm() {
    fileInput.value = '';
    selectedFiles = null;
    hideFileInfo();
    submitButtons.forEach(btn => btn.disabled = true);
    hideStatusMessage();
}

// Clear current file and allow new upload
function clearCurrentFile() {
    resetForm();
    showStatusMessage('File cleared. You can now upload a new file.', 'success');
    setTimeout(() => {
        hideStatusMessage();
    }, 2000);
}

// Handle drag and drop functionality
const container = document.querySelector('.container');

container.addEventListener('dragover', function(e) {
    e.preventDefault();
    container.style.backgroundColor = '#f0f8ff';
});

container.addEventListener('dragleave', function(e) {
    e.preventDefault();
    container.style.backgroundColor = 'white';
});

container.addEventListener('drop', function(e) {
    e.preventDefault();
    container.style.backgroundColor = 'white';
    
    const files = e.dataTransfer.files;
    if (files.length > 0) {
        // Update the file input
        fileInput.files = files;
        
        // Trigger the change event
        const event = new Event('change', { bubbles: true });
        fileInput.dispatchEvent(event);
    }
});

// Play notes sequence with their durations
async function playNotesSequence(durations, notes, soundFolder = CONFIG.AUDIO_FOLDERS.DRUMS) {
    console.log('Playing notes sequence:', { durations, notes });
    
    // Show playback info
    const playbackInfo = document.getElementById('playbackInfo');
    const currentNote = document.getElementById('currentNote');
    const progressBar = document.getElementById('progressBar');
    
    playbackInfo.style.display = 'block';
    
    // Create audio context if not exists
    if (!window.audioContext) {
        window.audioContext = new (window.AudioContext || window.webkitAudioContext)();
    }
    
    // Preload all audio files
    const audioBuffers = {};
    const uniqueNotes = [...new Set(notes)];
    
    try {
        // Load all unique note files
        for (const note of uniqueNotes) {
            const response = await fetch(`${soundFolder}/${note}.wav`);
            if (!response.ok) {
                throw new Error(`Failed to load ${note}.wav from ${soundFolder} folder`);
            }
            const arrayBuffer = await response.arrayBuffer();
            audioBuffers[note] = await window.audioContext.decodeAudioData(arrayBuffer);
            console.log(`Loaded audio for note: ${note}`);
        }
        
        // Calculate total duration
        const totalDuration = durations.reduce((sum, duration) => sum + duration, 0);
        
        // Play notes sequentially with visual feedback
        let currentTime = 0;
        
        for (let i = 0; i < notes.length; i++) {
            const note = notes[i];
            const duration = durations[i];
            
            // Schedule the note to play at the correct time
            scheduleNote(audioBuffers[note], currentTime, duration);
            
            // Schedule visual updates
            setTimeout(() => {
                currentNote.textContent = note;
                const progress = ((currentTime + duration) / totalDuration) * 100;
                progressBar.style.width = `${progress}%`;
            }, currentTime * 1000);
            
            // Add the duration to current time for next note
            currentTime += duration;
        }
        
        // Wait for all notes to finish playing
        await new Promise(resolve => setTimeout(resolve, currentTime * 1000 + 500));
        
        // Hide playback info
        playbackInfo.style.display = 'none';
        progressBar.style.width = '0%';
        
    } catch (error) {
        console.error('Error playing notes:', error);
        showStatusMessage('Error playing notes. Make sure drum files are available.', 'error');
        playbackInfo.style.display = 'none';
    }
}

// Schedule a single note to play at a specific time
function scheduleNote(audioBuffer, startTime, duration) {
    const source = window.audioContext.createBufferSource();
    source.buffer = audioBuffer;
    source.connect(window.audioContext.destination);
    
    // Start the note at the specified time
    source.start(window.audioContext.currentTime + startTime);
    
    // Optional: Stop the note after its duration (uncomment if needed)
    // source.stop(window.audioContext.currentTime + startTime + duration);
    
    console.log(`Scheduled note at time ${startTime}s for duration ${duration}s`);
}

// Alternative simpler implementation using HTML5 Audio (fallback)
async function playNotesSequenceSimple(durations, notes, soundFolder = CONFIG.AUDIO_FOLDERS.DRUMS) {
    console.log('Playing notes sequence (simple):', { durations, notes });
    
    try {
        for (let i = 0; i < notes.length; i++) {
            const note = notes[i];
            const duration = durations[i];
            
            // Create and play audio element
            const audio = new Audio(`${soundFolder}/${note}.wav`);
            audio.volume = 0.7;
            
            // Play the note
            await audio.play();
            console.log(`Playing note: ${note} for ${duration}s`);
            
            // Wait for the duration before playing next note
            await new Promise(resolve => setTimeout(resolve, duration * 1000));
        }
    } catch (error) {
        console.error('Error playing notes (simple):', error);
        showStatusMessage('Error playing notes. Make sure drum files are available.', 'error');
    }
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('File Upload Application initialized');
    hideStatusMessage();
    hideFileInfo();
});
