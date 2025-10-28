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
        // Check if we have either uploaded files or recorded audio
        const recordTab = document.getElementById('recordTab');
        const hasUploadedFiles = selectedFiles && selectedFiles.length > 0;
        const hasRecordedAudio = recordTab && recordTab.classList.contains('active') && recordedAudioBlob;
        
        if (!hasUploadedFiles && !hasRecordedAudio) {
            showStatusMessage('No files selected or recorded! Please upload a file or record audio.', 'error');
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

// Process the selected files or recorded audio
async function processFiles(files, apiEndpoint, buttonText, buttonId) {
    try {
        let file;
        
        // Check if we're using recorded audio or uploaded files
        const recordTab = document.getElementById('recordTab');
        if (recordTab && recordTab.classList.contains('active') && recordedAudioBlob) {
            // Use recorded audio
            console.log('Using recorded audio for processing');
            
            // Determine file extension based on MIME type
            let extension = 'webm'; // default
            if (recordedAudioBlob.type.includes('wav')) {
                extension = 'wav';
            } else if (recordedAudioBlob.type.includes('mp4')) {
                extension = 'mp4';
            } else if (recordedAudioBlob.type.includes('webm')) {
                extension = 'webm';
            }
            
            file = new File([recordedAudioBlob], `recorded_audio.${extension}`, { 
                type: recordedAudioBlob.type 
            });
            console.log(`Created file: recorded_audio.${extension}, type: ${recordedAudioBlob.type}`);
        } else if (files && files.length > 0) {
            // Use uploaded file
            file = files[0];
        } else {
            throw new Error('No audio file available for processing');
        }
        
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
            console.log(`Total notes received: ${result.notes.length}, Total durations: ${result.duration.length}`);
            
            // First play the audio sequence
            showStatusMessage(
                `${buttonText} completed! Playing audio sequence...`, 
                'success'
            );
            
            // Play the notes sequence (original functionality)
            await playNotesSequence(result.duration, result.notes, buttonText);
            
            // Then generate files for download
            showStatusMessage(
                `${buttonText} - Generating MIDI and WAV files...`, 
                'success'
            );
            
            // Generate files but store them, don't auto-download
            const midiBlob = await generateMIDIFile(result.duration, result.notes, buttonText);
            const midiFilename = `${buttonText.replace(/\s+/g, '_').toLowerCase()}_output.mid`;
            
            let wavBlob = null;
            let wavFilename = null;
            
            // Generate WAV by combining actual audio files
            try {
                console.log('Generating WAV by combining actual audio files');
                wavBlob = await generateWAVFromAudioFiles(result.duration, result.notes, buttonText);
                wavFilename = `${buttonText.replace(/\s+/g, '_').toLowerCase()}_combined_output.wav`;
            } catch (wavError) {
                console.error('WAV generation failed:', wavError);
                // Fallback to synthesized WAV
                try {
                    console.log('Fallback to synthetic WAV');
                    wavBlob = await generateWAVFile(result.duration, result.notes, buttonText);
                    wavFilename = `${buttonText.replace(/\s+/g, '_').toLowerCase()}_synthetic_output.wav`;
                } catch (synthError) {
                    console.error('Synthetic WAV also failed:', synthError);
                }
            }
            
            // Show download buttons
            showDownloadButtons(midiBlob, midiFilename, wavBlob, wavFilename);
            
            showStatusMessage(
                `${buttonText} completed successfully! Audio played. Files ready for download. You can perform other operations on the same file.`, 
                'success'
            );
            
            // Re-enable all submit buttons for additional operations
            setTimeout(() => {
                submitButtons.forEach(btn => btn.disabled = false);
            }, 500);
        } else if (result.midi_data || result.midi_url) {
            // Handle direct MIDI response from backend
            showStatusMessage(
                `${buttonText} completed! Downloading MIDI file...`, 
                'success'
            );
            
            if (result.midi_url) {
                // Download MIDI from URL
                downloadMIDIFromURL(result.midi_url, `${buttonText.replace(/\s+/g, '_').toLowerCase()}_output.mid`);
            } else if (result.midi_data) {
                // Handle base64 MIDI data
                downloadMIDIFromBase64(result.midi_data, `${buttonText.replace(/\s+/g, '_').toLowerCase()}_output.mid`);
            }
            
            showStatusMessage(
                `${buttonText} completed successfully! MIDI file downloaded. You can perform other operations on the same file.`, 
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

// Generate MIDI file from notes and durations
async function generateMIDIFile(durations, notes, buttonText) {
    console.log('Generating MIDI file:', { durations, notes, buttonText });
    
    // Show MIDI generation info
    const playbackInfo = document.getElementById('playbackInfo');
    const currentNote = document.getElementById('currentNote');
    const progressBar = document.getElementById('progressBar');
    
    playbackInfo.style.display = 'block';
    currentNote.textContent = 'Generating MIDI...';
    progressBar.style.width = '50%';
    
    try {
        // Create basic MIDI file structure
        const midiData = createMIDIFromNotes(durations, notes, buttonText);
        
        // Update progress
        progressBar.style.width = '100%';
        currentNote.textContent = 'MIDI Ready!';
        
        // Hide generation info after a short delay
        setTimeout(() => {
            playbackInfo.style.display = 'none';
            progressBar.style.width = '0%';
        }, 1000);
        
        return midiData;
        
    } catch (error) {
        console.error('Error generating MIDI:', error);
        playbackInfo.style.display = 'none';
        throw error;
    }
}

// Create MIDI file from notes and durations
function createMIDIFromNotes(durations, notes, buttonText) {
    // MIDI file header (format 0, 1 track, 480 ticks per quarter note)
    const headerChunk = new Uint8Array([
        0x4D, 0x54, 0x68, 0x64, // "MThd" header
        0x00, 0x00, 0x00, 0x06, // Header length (6 bytes)
        0x00, 0x00,             // Format 0
        0x00, 0x01,             // 1 track
        0x01, 0xE0              // 480 ticks per quarter note
    ]);
    
    // Create MIDI events for each note
    const midiEvents = [];
    let currentTick = 0;
    const ticksPerSecond = 480; // 480 ticks per quarter note, 120 BPM
    
    console.log(`Creating MIDI events for ${notes.length} notes:`, notes);
    console.log('Durations:', durations);
    
    // Add tempo event (120 BPM = 500000 microseconds per quarter note)
    midiEvents.push([0, 0xFF, 0x51, 0x03, 0x07, 0xA1, 0x20]); // Tempo meta event
    
    // Map note names to MIDI note numbers
    const noteMapping = getNoteMapping(buttonText);
    console.log('Note mapping:', noteMapping);
    
    for (let i = 0; i < notes.length; i++) {
        const note = notes[i];
        const duration = Math.max(durations[i] || 0.5, 0.1); // Ensure minimum duration
        const midiNote = noteMapping[note] || 60; // Default to middle C if note not found
        const velocity = 80; // Note velocity (volume)
        
        console.log(`Note ${i + 1}: ${note} -> MIDI ${midiNote}, duration: ${duration}s`);
        
        // Calculate duration in ticks (ensure minimum duration)
        const durationTicks = Math.max(Math.round(duration * ticksPerSecond), 48); // Minimum 1/10 second
        
        // Delta time for note on (gap between previous note and this one)
        const deltaTime = encodeVariableLength(i === 0 ? 0 : 0);
        
        // Note on event (channel 9 for percussion)
        midiEvents.push([...deltaTime, 0x99, midiNote, velocity]); // Note on percussion channel
        
        // Note off event after the duration
        const noteDuration = encodeVariableLength(durationTicks);
        midiEvents.push([...noteDuration, 0x89, midiNote, 0x40]); // Note off percussion channel
        
        currentTick += durationTicks;
    }
    
    console.log(`Total MIDI events created: ${midiEvents.length}, Total ticks: ${currentTick}`);
    console.log(`Estimated total duration: ${(currentTick / ticksPerSecond).toFixed(2)} seconds`);
    
    // Add end of track event
    midiEvents.push([0x00, 0xFF, 0x2F, 0x00]);
    
    // Convert events to bytes
    const trackData = new Uint8Array(midiEvents.flat());
    
    // Track chunk header
    const trackHeader = new Uint8Array([
        0x4D, 0x54, 0x72, 0x6B, // "MTrk" header
        ...intToBytes(trackData.length, 4) // Track length
    ]);
    
    // Combine header, track header, and track data
    const midiFile = new Uint8Array(headerChunk.length + trackHeader.length + trackData.length);
    midiFile.set(headerChunk, 0);
    midiFile.set(trackHeader, headerChunk.length);
    midiFile.set(trackData, headerChunk.length + trackHeader.length);
    
    return new Blob([midiFile], { type: 'audio/midi' });
}

// Get MIDI note mapping based on instrument type
function getNoteMapping(buttonText) {
    if (buttonText.toLowerCase().includes('tabla')) {
        // Tabla note mapping
        return {
            'Dha': 36,  // Bass drum
            'Dhin': 38, // Snare
            'Ta': 42,   // Closed hi-hat
            'Tin': 44,  // Pedal hi-hat
            'Na': 46,   // Open hi-hat
            'Ghe': 41,  // Low tom
            'Ka': 43,   // High tom
            'Tun': 45,  // Mid tom
        };
    } else {
        // Drum kit mapping
        return {
            'Dha': 36,  // Bass drum
            'Dhin': 38, // Snare
            'Ta': 42,   // Closed hi-hat
            'Tin': 44,  // Pedal hi-hat
            'Na': 46,   // Open hi-hat
            'Ghe': 41,  // Low tom
            'Ka': 43,   // High tom
            'Tun': 45,  // Mid tom
            'kick': 36, // Bass drum
            'snare': 38, // Snare
            'hihat': 42, // Hi-hat
        };
    }
}

// Encode variable length quantity (MIDI format)
function encodeVariableLength(value) {
    const bytes = [];
    bytes.push(value & 0x7F);
    value >>= 7;
    while (value > 0) {
        bytes.unshift((value & 0x7F) | 0x80);
        value >>= 7;
    }
    return bytes;
}

// Convert integer to bytes
function intToBytes(value, byteCount) {
    const bytes = [];
    for (let i = byteCount - 1; i >= 0; i--) {
        bytes.push((value >> (i * 8)) & 0xFF);
    }
    return bytes;
}

// Download MIDI file
function downloadMIDIFile(midiBlob, filename) {
    const url = URL.createObjectURL(midiBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log(`Downloaded MIDI file: ${filename}`);
}

// Download MIDI from URL (if backend provides URL)
async function downloadMIDIFromURL(midiUrl, filename) {
    try {
        const response = await fetch(midiUrl);
        if (!response.ok) {
            throw new Error('Failed to fetch MIDI file from URL');
        }
        const midiBlob = await response.blob();
        downloadMIDIFile(midiBlob, filename);
    } catch (error) {
        console.error('Error downloading MIDI from URL:', error);
        showStatusMessage('Error downloading MIDI file.', 'error');
    }
}

// Download MIDI from base64 data
function downloadMIDIFromBase64(base64Data, filename) {
    try {
        const binaryString = atob(base64Data);
        const bytes = new Uint8Array(binaryString.length);
        for (let i = 0; i < binaryString.length; i++) {
            bytes[i] = binaryString.charCodeAt(i);
        }
        const midiBlob = new Blob([bytes], { type: 'audio/midi' });
        downloadMIDIFile(midiBlob, filename);
    } catch (error) {
        console.error('Error converting base64 to MIDI:', error);
        showStatusMessage('Error processing MIDI data.', 'error');
    }
}

// Generate WAV file from notes and durations
async function generateWAVFile(durations, notes, buttonText) {
    console.log('Generating WAV file:', { durations, notes, buttonText });
    
    // Show WAV generation info
    const playbackInfo = document.getElementById('playbackInfo');
    const currentNote = document.getElementById('currentNote');
    const progressBar = document.getElementById('progressBar');
    
    playbackInfo.style.display = 'block';
    currentNote.textContent = 'Generating WAV...';
    progressBar.style.width = '25%';
    
    try {
        // Check for Web Audio API support
        if (!window.AudioContext && !window.webkitAudioContext) {
            throw new Error('Web Audio API not supported in this browser');
        }
        
        // Audio context for synthesis (fixed sample rate)
        let audioContext;
        try {
            audioContext = new (window.AudioContext || window.webkitAudioContext)({ sampleRate: 44100 });
        } catch (e) {
            console.warn('Failed to create AudioContext with specified sample rate, using default:', e);
            audioContext = new (window.AudioContext || window.webkitAudioContext)();
        }
        
        // Resume context if suspended (required by browsers)
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
        
        const sampleRate = audioContext.sampleRate || 44100;
        
        // Calculate total duration with proper timing
        let totalDuration = durations.reduce((sum, duration) => sum + Math.max(duration, 0.1), 0) + 1; // Add 1s buffer
        const totalSamples = Math.floor(totalDuration * sampleRate);
        const audioBuffer = audioContext.createBuffer(1, totalSamples, sampleRate);
        const channelData = audioBuffer.getChannelData(0);
        
        progressBar.style.width = '50%';
        currentNote.textContent = 'Synthesizing percussion sounds...';
        
        let currentTime = 0;
        
        for (let i = 0; i < notes.length; i++) {
            const note = notes[i];
            const duration = Math.max(durations[i] || 0.5, 0.1); // Minimum 0.1s per note
            
            console.log(`Processing note ${i + 1}/${notes.length}: ${note} (${duration}s)`);
            currentNote.textContent = `Processing: ${note} (${i + 1}/${notes.length})`;
            
            // Generate percussion sound based on note type
            const noteSound = generatePercussionSound(note, duration, sampleRate, buttonText);
            const startSample = Math.floor(currentTime * sampleRate);
            
            // Mix the note sound into the main buffer
            for (let j = 0; j < noteSound.length && (startSample + j) < totalSamples; j++) {
                channelData[startSample + j] += noteSound[j] * 0.3; // Volume scaling
            }
            
            currentTime += duration;
            progressBar.style.width = `${50 + (i / notes.length) * 40}%`;
        }
        
        // Normalize audio to prevent clipping
        let maxAmplitude = Math.max(...Array.from(channelData).map(Math.abs));
        if (maxAmplitude > 0.95) {
            const normalizeGain = 0.95 / maxAmplitude;
            for (let i = 0; i < channelData.length; i++) {
                channelData[i] *= normalizeGain;
            }
        }
        
        progressBar.style.width = '95%';
        currentNote.textContent = 'Converting to WAV...';
        
        // Convert to WAV format
        const wavData = audioBufferToWav(audioBuffer);
        const wavBlob = new Blob([wavData], { type: 'audio/wav' });
        
        progressBar.style.width = '100%';
        currentNote.textContent = 'WAV Ready!';
        
        // Hide generation info after a short delay
        setTimeout(() => {
            playbackInfo.style.display = 'none';
            progressBar.style.width = '0%';
        }, 1500);
        
        console.log(`WAV generated: ${notes.length} notes, ${totalDuration.toFixed(2)}s duration`);
        return wavBlob;
        
    } catch (error) {
        console.error('Error generating WAV with Web Audio API:', error);
        playbackInfo.style.display = 'none';
        
        // Try fallback method
        try {
            console.log('Attempting fallback WAV generation...');
            return generateSimpleWAVFile(durations, notes, buttonText);
        } catch (fallbackError) {
            console.error('Fallback WAV generation also failed:', fallbackError);
            throw new Error(`WAV generation failed: ${error.message}`);
        }
    }
}

// Simple WAV generation fallback (no Web Audio API)
function generateSimpleWAVFile(durations, notes, buttonText) {
    console.log('Generating simple WAV file without Web Audio API');
    
    const sampleRate = 44100;
    const duration = durations.reduce((sum, d) => sum + Math.max(d || 0.5, 0.1), 0);
    const numSamples = Math.floor(duration * sampleRate);
    const audioData = new Float32Array(numSamples);
    
    let currentTime = 0;
    
    // Generate simple beep tones for each note
    for (let i = 0; i < notes.length; i++) {
        const note = notes[i];
        const noteDuration = Math.max(durations[i] || 0.5, 0.1);
        const startSample = Math.floor(currentTime * sampleRate);
        const endSample = Math.min(startSample + Math.floor(noteDuration * sampleRate), numSamples);
        
        // Simple frequency mapping
        let frequency = 440; // Default A4
        if (note.toLowerCase().includes('dha') || note.toLowerCase().includes('kick')) {
            frequency = 80;
        } else if (note.toLowerCase().includes('dhin') || note.toLowerCase().includes('snare')) {
            frequency = 200;
        } else if (note.toLowerCase().includes('ta') || note.toLowerCase().includes('hihat')) {
            frequency = 800;
        } else {
            frequency = 400;
        }
        
        // Generate sine wave with envelope
        for (let j = startSample; j < endSample; j++) {
            const t = (j - startSample) / sampleRate;
            const envelope = Math.exp(-t * 5); // Quick decay
            audioData[j] += Math.sin(2 * Math.PI * frequency * t) * envelope * 0.3;
        }
        
        currentTime += noteDuration;
    }
    
    // Convert to WAV
    return simpleAudioToWav(audioData, sampleRate);
}

// Simple audio to WAV converter
function simpleAudioToWav(audioData, sampleRate) {
    const length = audioData.length;
    const arrayBuffer = new ArrayBuffer(44 + length * 2);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true); // PCM
    view.setUint16(22, 1, true); // Mono
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * 2, true);
    view.setUint16(32, 2, true);
    view.setUint16(34, 16, true);
    writeString(36, 'data');
    view.setUint32(40, length * 2, true);
    
    // Convert float samples to 16-bit PCM
    let offset = 44;
    for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, audioData[i]));
        view.setInt16(offset, sample * 0x7FFF, true);
        offset += 2;
    }
    
    return new Blob([arrayBuffer], { type: 'audio/wav' });
}

// Generate percussion sound based on note name
function generatePercussionSound(noteName, duration, sampleRate, buttonText) {
    const numSamples = Math.floor(duration * sampleRate);
    const sound = new Float32Array(numSamples);
    
    // Define percussion characteristics based on note
    let freq, attackTime, decayTime, noiseAmount;
    
    // Map note names to percussion characteristics
    const noteMapping = getNoteMapping(buttonText);
    const midiNote = noteMapping[noteName] || 60;
    
    // Percussion sound parameters based on MIDI note
    if (midiNote <= 38) {
        // Bass/kick sounds (Dha, Dhin)
        freq = 60 + Math.random() * 40;
        attackTime = 0.005;
        decayTime = 0.3;
        noiseAmount = 0.4;
    } else if (midiNote <= 42) {
        // Mid-range percussion (Ta, hihat)
        freq = 200 + Math.random() * 300;
        attackTime = 0.002;
        decayTime = 0.15;
        noiseAmount = 0.7;
    } else {
        // High percussion (Tin, Na, Tun, etc.)
        freq = 400 + Math.random() * 600;
        attackTime = 0.001;
        decayTime = 0.2;
        noiseAmount = 0.6;
    }
    
    // Generate the sound
    for (let i = 0; i < numSamples; i++) {
        const time = i / sampleRate;
        const normalizedTime = time / duration;
        
        // Envelope (attack and exponential decay)
        let envelope = 1;
        if (time < attackTime) {
            envelope = time / attackTime;
        } else {
            envelope = Math.exp(-(time - attackTime) / decayTime);
        }
        
        // Generate sound (combination of sine wave and noise)
        const sineWave = Math.sin(2 * Math.PI * freq * time * (1 + normalizedTime * 0.5));
        const noise = (Math.random() * 2 - 1) * noiseAmount;
        const toneComponent = sineWave * (1 - noiseAmount);
        
        sound[i] = (toneComponent + noise) * envelope;
        
        // Add some frequency modulation for more realistic percussion
        freq *= (1 - normalizedTime * 0.1);
    }
    
    return sound;
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(audioBuffer) {
    const numChannels = audioBuffer.numberOfChannels;
    const sampleRate = audioBuffer.sampleRate;
    const format = 1; // PCM
    const bitDepth = 16;
    
    const bytesPerSample = bitDepth / 8;
    const blockAlign = numChannels * bytesPerSample;
    
    const buffer = audioBuffer.getChannelData(0);
    const length = buffer.length;
    const arrayBuffer = new ArrayBuffer(44 + length * bytesPerSample);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * bytesPerSample, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, format, true);
    view.setUint16(22, numChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * blockAlign, true);
    view.setUint16(32, blockAlign, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length * bytesPerSample, true);
    
    // Convert float samples to 16-bit PCM
    let offset = 44;
    for (let i = 0; i < length; i++) {
        const sample = Math.max(-1, Math.min(1, buffer[i]));
        view.setInt16(offset, sample * 0x7FFF, true);
        offset += 2;
    }
    
    return arrayBuffer;
}

// Download WAV file
function downloadWAVFile(wavBlob, filename) {
    const url = URL.createObjectURL(wavBlob);
    const a = document.createElement('a');
    a.href = url;
    a.download = filename;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
    console.log(`Downloaded WAV file: ${filename}`);
}

// Generate WAV by loading and combining actual audio files
async function generateWAVFromAudioFiles(durations, notes, buttonText) {
    console.log('Combining actual audio files for WAV:', { durations, notes, buttonText });
    
    const audioFolder = buttonText.toLowerCase().includes('drum') ? CONFIG.AUDIO_FOLDERS.DRUMS : CONFIG.AUDIO_FOLDERS.TABLA;
    const audioContext = new (window.AudioContext || window.webkitAudioContext)();
    if (audioContext.state === 'suspended') {
        await audioContext.resume();
    }
    
    const sampleRate = audioContext.sampleRate;
    const audioBuffers = [];
    const actualDurations = [];
    
    // Load all audio files
    for (let i = 0; i < notes.length; i++) {
        const note = notes[i];
        const duration = Math.max(durations[i] || 0.5, 0.1);
        
        console.log(`Loading audio file for note: ${note}`);
        
        try {
            const audioBuffer = await loadAudioFile(note, audioFolder, audioContext);
            if (audioBuffer) {
                audioBuffers.push(audioBuffer);
                // Use the actual file duration or specified duration, whichever is smaller
                const actualDuration = Math.min(audioBuffer.duration, duration);
                actualDurations.push(actualDuration);
                console.log(`Loaded ${note}: ${actualDuration.toFixed(2)}s`);
            } else {
                throw new Error(`Could not load audio for ${note}`);
            }
        } catch (error) {
            console.warn(`Failed to load audio for ${note}:`, error);
            // Use synthetic sound as fallback
            const syntheticBuffer = await generateSyntheticAudioBuffer(note, duration, audioContext);
            audioBuffers.push(syntheticBuffer);
            actualDurations.push(duration);
        }
    }
    
    // Calculate total duration and create combined buffer
    const totalDuration = actualDurations.reduce((sum, dur) => sum + dur, 0);
    const totalSamples = Math.floor(totalDuration * sampleRate);
    const combinedBuffer = audioContext.createBuffer(1, totalSamples, sampleRate);
    const outputData = combinedBuffer.getChannelData(0);
    
    // Combine all audio buffers sequentially
    let currentSample = 0;
    
    for (let i = 0; i < audioBuffers.length; i++) {
        const buffer = audioBuffers[i];
        const duration = actualDurations[i];
        const durationSamples = Math.floor(duration * sampleRate);
        
        // Get source data (convert to mono if needed)
        const sourceData = buffer.numberOfChannels === 1 ? 
            buffer.getChannelData(0) : 
            buffer.getChannelData(0); // Take left channel if stereo
        
        // Copy samples (with resampling if needed)
        const sourceSamples = Math.min(sourceData.length, durationSamples);
        for (let j = 0; j < sourceSamples && (currentSample + j) < totalSamples; j++) {
            outputData[currentSample + j] = sourceData[j];
        }
        
        currentSample += durationSamples;
        console.log(`Combined ${notes[i]}: ${sourceSamples} samples at position ${currentSample - durationSamples}`);
    }
    
    // Convert to WAV
    const wavData = audioBufferToWav(combinedBuffer);
    console.log(`Generated combined WAV: ${notes.length} notes, ${totalDuration.toFixed(2)}s total`);
    
    return new Blob([wavData], { type: 'audio/wav' });
}

// Load audio file and return AudioBuffer
async function loadAudioFile(noteName, audioFolder, audioContext) {
    const audioFileName = `${noteName.toLowerCase()}.wav`;
    let audioPath = `${audioFolder}/${audioFileName}`;
    
    try {
        // Try lowercase first
        let response = await fetch(audioPath);
        if (!response.ok) {
            // Try capitalized
            const capitalizedFileName = noteName.charAt(0).toUpperCase() + noteName.slice(1).toLowerCase() + '.wav';
            audioPath = `${audioFolder}/${capitalizedFileName}`;
            response = await fetch(audioPath);
            if (!response.ok) {
                throw new Error(`Audio file not found: ${audioPath}`);
            }
        }
        
        console.log(`Successfully loaded: ${audioPath}`);
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        return audioBuffer;
        
    } catch (error) {
        console.error(`Failed to load audio file for ${noteName}:`, error);
        return null;
    }
}

// Generate synthetic audio buffer as fallback
async function generateSyntheticAudioBuffer(noteName, duration, audioContext) {
    const sampleRate = audioContext.sampleRate;
    const numSamples = Math.floor(duration * sampleRate);
    const audioBuffer = audioContext.createBuffer(1, numSamples, sampleRate);
    const channelData = audioBuffer.getChannelData(0);
    
    // Simple frequency mapping
    let frequency = 440;
    if (noteName.toLowerCase().includes('dha')) frequency = 80;
    else if (noteName.toLowerCase().includes('dhin')) frequency = 120;
    else if (noteName.toLowerCase().includes('ta')) frequency = 800;
    else if (noteName.toLowerCase().includes('tin')) frequency = 600;
    else if (noteName.toLowerCase().includes('na')) frequency = 1000;
    
    // Generate sine wave with envelope
    for (let i = 0; i < numSamples; i++) {
        const time = i / sampleRate;
        const envelope = Math.exp(-time * 3); // Decay
        channelData[i] = Math.sin(2 * Math.PI * frequency * time) * envelope * 0.3;
    }
    
    console.log(`Generated synthetic audio for ${noteName}: ${frequency}Hz, ${duration}s`);
    return audioBuffer;
}

// Show download buttons for generated files
function showDownloadButtons(midiBlob, midiFilename, wavBlob, wavFilename) {
    // Create or get download section
    let downloadSection = document.getElementById('downloadSection');
    if (!downloadSection) {
        downloadSection = document.createElement('div');
        downloadSection.id = 'downloadSection';
        downloadSection.className = 'download-section';
        
        // Insert after playback info
        const playbackInfo = document.getElementById('playbackInfo');
        playbackInfo.parentNode.insertBefore(downloadSection, playbackInfo.nextSibling);
    }
    
    // Clear existing buttons
    downloadSection.innerHTML = '';
    
    const title = document.createElement('h3');
    title.textContent = '📥 Download Files';
    title.style.marginBottom = '15px';
    downloadSection.appendChild(title);
    
    const buttonContainer = document.createElement('div');
    buttonContainer.className = 'download-buttons';
    
    // MIDI download button
    const midiButton = document.createElement('button');
    midiButton.className = 'download-btn midi-btn';
    midiButton.innerHTML = '🎼 Download MIDI';
    midiButton.onclick = () => downloadMIDIFile(midiBlob, midiFilename);
    buttonContainer.appendChild(midiButton);
    
    // WAV download button (if available)
    if (wavBlob) {
        const wavButton = document.createElement('button');
        wavButton.className = 'download-btn wav-btn';
        wavButton.innerHTML = '🔊 Download WAV';
        wavButton.onclick = () => downloadWAVFile(wavBlob, wavFilename);
        buttonContainer.appendChild(wavButton);
    }
    
    downloadSection.appendChild(buttonContainer);
    downloadSection.style.display = 'block';
    
    // Add CSS if not already present
    if (!document.getElementById('downloadSectionStyles')) {
        const style = document.createElement('style');
        style.id = 'downloadSectionStyles';
        style.textContent = `
            .download-section {
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                border-radius: 15px;
                padding: 20px;
                margin: 20px 0;
                text-align: center;
                color: white;
                box-shadow: 0 8px 32px rgba(0,0,0,0.1);
            }
            .download-buttons {
                display: flex;
                gap: 15px;
                justify-content: center;
                flex-wrap: wrap;
            }
            .download-btn {
                background: rgba(255,255,255,0.2);
                border: 2px solid rgba(255,255,255,0.3);
                border-radius: 10px;
                color: white;
                padding: 12px 24px;
                font-size: 16px;
                font-weight: bold;
                cursor: pointer;
                transition: all 0.3s ease;
                backdrop-filter: blur(10px);
            }
            .download-btn:hover {
                background: rgba(255,255,255,0.3);
                border-color: rgba(255,255,255,0.5);
                transform: translateY(-2px);
                box-shadow: 0 5px 15px rgba(0,0,0,0.2);
            }
            .midi-btn:hover { background: rgba(255,215,0,0.3); }
            .wav-btn:hover { background: rgba(50,205,50,0.3); }
        `;
        document.head.appendChild(style);
    }
}

// Original audio playback functions with recording capability
async function playNotesSequence(durations, notes, buttonText) {
    console.log('Playing and recording notes sequence:', { durations, notes, buttonText });
    
    // Show playback info
    const playbackInfo = document.getElementById('playbackInfo');
    const currentNote = document.getElementById('currentNote');
    const progressBar = document.getElementById('progressBar');
    
    playbackInfo.style.display = 'block';
    
    const audioFolder = buttonText.toLowerCase().includes('drum') ? CONFIG.AUDIO_FOLDERS.DRUMS : CONFIG.AUDIO_FOLDERS.TABLA;
    
    // Simple playback without recording complexity
    window.currentRecordingContext = null;
    
    let totalDuration = durations.reduce((sum, duration) => sum + duration, 0);
    let elapsedTime = 0;
    
    for (let i = 0; i < notes.length; i++) {
        const note = notes[i];
        const duration = durations[i];
        
        currentNote.textContent = `Playing: ${note} (${i + 1}/${notes.length})`;
        
        // Update progress bar
        const progress = (elapsedTime / totalDuration) * 100;
        progressBar.style.width = `${progress}%`;
        
        // Play the note (with recording if available)
        await playNoteSound(note, audioFolder);
        
        // Wait for the duration of this note
        await new Promise(resolve => setTimeout(resolve, duration * 1000));
        
        elapsedTime += duration;
    }
    
    // Playback completed
    
    // Complete progress
    progressBar.style.width = '100%';
    currentNote.textContent = 'Playback Complete!';
    
    // Hide playback info after a short delay
    setTimeout(() => {
        playbackInfo.style.display = 'none';
        progressBar.style.width = '0%';
    }, 1000);
}

// Play individual note sound with recording capability
async function playNoteSound(noteName, audioFolder) {
    try {
        // Try to find and play the audio file
        const audioFileName = `${noteName.toLowerCase()}.wav`;
        const audioPath = `${audioFolder}/${audioFileName}`;
        
        console.log(`Attempting to play: ${audioPath}`);
        
        // Check if we have a recording context
        const recordingContext = window.currentRecordingContext;
        
        if (recordingContext) {
            // Use Web Audio API for playback that can be recorded
            return await playNoteWithWebAudio(audioPath, noteName, recordingContext);
        } else {
            // Fallback to regular HTML audio
            const audio = new Audio(audioPath);
            audio.volume = 0.7;
            
            return new Promise((resolve, reject) => {
                audio.onended = resolve;
                audio.onerror = () => {
                    console.warn(`Audio file not found: ${audioPath}, trying capitalized name`);
                    // Try capitalized version
                    const capitalizedPath = audioPath.replace(/([^/]+)\.wav$/, (match, name) => {
                        return name.charAt(0).toUpperCase() + name.slice(1) + '.wav';
                    });
                    console.log(`Trying capitalized path: ${capitalizedPath}`);
                    const capitalizedAudio = new Audio(capitalizedPath);
                    capitalizedAudio.volume = 0.7;
                    capitalizedAudio.onended = resolve;
                    capitalizedAudio.onerror = () => {
                        console.warn(`Capitalized audio file also not found: ${capitalizedPath}, using Web Audio API`);
                        playSystemSound(noteName).then(resolve).catch(resolve);
                    };
                    capitalizedAudio.play().catch(() => {
                        console.warn(`Could not play capitalized audio file: ${capitalizedPath}, using Web Audio API`);
                        playSystemSound(noteName).then(resolve).catch(resolve);
                    });
                };
                audio.play().catch(() => {
                    console.warn(`Could not play audio file: ${audioPath}, trying capitalized name`);
                    // Try capitalized version
                    const capitalizedPath = audioPath.replace(/([^/]+)\.wav$/, (match, name) => {
                        return name.charAt(0).toUpperCase() + name.slice(1) + '.wav';
                    });
                    const capitalizedAudio = new Audio(capitalizedPath);
                    capitalizedAudio.volume = 0.7;
                    capitalizedAudio.onended = resolve;
                    capitalizedAudio.onerror = () => {
                        console.warn(`All audio options failed, using Web Audio API`);
                        playSystemSound(noteName).then(resolve).catch(resolve);
                    };
                    capitalizedAudio.play().catch(() => {
                        console.warn(`All audio options failed, using Web Audio API`);
                        playSystemSound(noteName).then(resolve).catch(resolve);
                    });
                });
            });
        }
    } catch (error) {
        console.warn('Error playing note sound:', error);
        return playSystemSound(noteName);
    }
}

// Play note using Web Audio API (for recording)
async function playNoteWithWebAudio(audioPath, noteName, recordingContext) {
    try {
        const { audioContext, dest } = recordingContext;
        
        // Try to load the audio file
        const response = await fetch(audioPath);
        if (!response.ok) {
            // Try with capitalized name if lowercase fails
            const capitalizedPath = audioPath.replace(/([^/]+)\.wav$/, (match, name) => {
                return name.charAt(0).toUpperCase() + name.slice(1) + '.wav';
            });
            console.log(`Trying capitalized path: ${capitalizedPath}`);
            const capitalizedResponse = await fetch(capitalizedPath);
            if (!capitalizedResponse.ok) throw new Error('Audio file not found');
            const arrayBuffer = await capitalizedResponse.arrayBuffer();
            const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
            return await playAudioBuffer(audioBuffer, audioContext, dest);
        }
        
        const arrayBuffer = await response.arrayBuffer();
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        return await playAudioBuffer(audioBuffer, audioContext, dest);
        
    } catch (error) {
        console.warn(`Could not load audio file ${audioPath}:`, error);
        // Fallback to synthesized sound with recording
        return await playSystemSoundWithRecording(noteName, recordingContext);
    }
}

// Helper function to play audio buffer
async function playAudioBuffer(audioBuffer, audioContext, dest) {
    // Create audio source
    const source = audioContext.createBufferSource();
    const gainNode = audioContext.createGain();
    
    source.buffer = audioBuffer;
    gainNode.gain.value = 0.7;
    
    // Connect to both speakers and recording
    source.connect(gainNode);
    gainNode.connect(audioContext.destination); // For hearing
    if (dest) {
        gainNode.connect(dest); // For recording (if available)
    }
    
    // Play the sound
    source.start();
    
    // Wait for it to finish
    return new Promise(resolve => {
        source.onended = resolve;
        setTimeout(resolve, audioBuffer.duration * 1000 + 100); // Fallback timeout
    });
}

// System sound with recording capability
async function playSystemSoundWithRecording(noteName, recordingContext) {
    try {
        const { audioContext, dest } = recordingContext;
        
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        // Map note to frequency
        let frequency = 440;
        if (noteName.toLowerCase().includes('dha')) frequency = 80;
        else if (noteName.toLowerCase().includes('dhin')) frequency = 120;
        else if (noteName.toLowerCase().includes('ta')) frequency = 800;
        else if (noteName.toLowerCase().includes('tin')) frequency = 600;
        else if (noteName.toLowerCase().includes('na')) frequency = 1000;
        
        oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
        oscillator.type = 'sawtooth';
        
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
        
        // Connect to both speakers and recording
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination); // For hearing
        gainNode.connect(dest); // For recording
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.3);
        
        return new Promise(resolve => {
            setTimeout(resolve, 300);
        });
    } catch (error) {
        console.error('Error with Web Audio API recording fallback:', error);
        return Promise.resolve();
    }
}

// Fallback system sound using Web Audio API (without recording)
async function playSystemSound(noteName) {
    try {
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        if (audioContext.state === 'suspended') {
            await audioContext.resume();
        }
        
        const oscillator = audioContext.createOscillator();
        const gainNode = audioContext.createGain();
        
        // Map note to frequency
        let frequency = 440;
        if (noteName.toLowerCase().includes('dha')) frequency = 80;
        else if (noteName.toLowerCase().includes('dhin')) frequency = 120;
        else if (noteName.toLowerCase().includes('ta')) frequency = 800;
        else if (noteName.toLowerCase().includes('tin')) frequency = 600;
        else if (noteName.toLowerCase().includes('na')) frequency = 1000;
        
        oscillator.frequency.setValueAtTime(frequency, audioContext.currentTime);
        oscillator.type = 'sawtooth';
        
        gainNode.gain.setValueAtTime(0.1, audioContext.currentTime);
        gainNode.gain.exponentialRampToValueAtTime(0.01, audioContext.currentTime + 0.3);
        
        oscillator.connect(gainNode);
        gainNode.connect(audioContext.destination);
        
        oscillator.start(audioContext.currentTime);
        oscillator.stop(audioContext.currentTime + 0.3);
        
        return new Promise(resolve => {
            setTimeout(resolve, 300);
        });
    } catch (error) {
        console.error('Error with Web Audio API fallback:', error);
        return Promise.resolve();
    }
}

// Audio Recording Variables
let mediaRecorder = null;
let recordedChunks = [];
let isRecording = false;
let recordingStartTime = null;
let recordingTimer = null;
let recordedAudioBlob = null;
let audioStream = null;

// Input method switching
function switchInputMethod(method) {
    const uploadTab = document.getElementById('uploadTab');
    const recordTab = document.getElementById('recordTab');
    const uploadSection = document.getElementById('uploadSection');
    const recordSection = document.getElementById('recordSection');
    
    if (method === 'upload') {
        uploadTab.classList.add('active');
        recordTab.classList.remove('active');
        uploadSection.classList.add('active');
        recordSection.classList.remove('active');
        
        // Clear any recording and use uploaded file
        if (selectedFiles) {
            enableSubmitButtons();
        }
    } else if (method === 'record') {
        uploadTab.classList.remove('active');
        recordTab.classList.add('active');
        uploadSection.classList.remove('active');
        recordSection.classList.add('active');
        
        // Check if we have a recorded file
        if (recordedAudioBlob) {
            enableSubmitButtons();
        } else {
            disableSubmitButtons();
        }
    }
}

// Start audio recording
async function startRecording() {
    try {
        console.log('Starting audio recording...');
        
        // Request microphone access
        audioStream = await navigator.mediaDevices.getUserMedia({ 
            audio: {
                echoCancellation: true,
                noiseSuppression: true,
                sampleRate: 44100
            } 
        });
        
        // Create MediaRecorder with better format support
        let options = {
            mimeType: 'audio/wav',
            audioBitsPerSecond: 128000
        };
        
        // Try different formats in order of preference
        if (!MediaRecorder.isTypeSupported(options.mimeType)) {
            options.mimeType = 'audio/webm;codecs=opus';
            if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                options.mimeType = 'audio/webm';
                if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                    options.mimeType = 'audio/mp4';
                    if (!MediaRecorder.isTypeSupported(options.mimeType)) {
                        // Fallback to default
                        options = {};
                    }
                }
            }
        }
        
        mediaRecorder = new MediaRecorder(audioStream, options);
        recordedChunks = [];
        
        mediaRecorder.ondataavailable = (event) => {
            if (event.data.size > 0) {
                recordedChunks.push(event.data);
            }
        };
        
        mediaRecorder.onstop = () => {
            console.log('Recording stopped, processing audio...');
            processRecordedAudio();
        };
        
        // Start recording
        mediaRecorder.start(1000); // Collect data every second
        isRecording = true;
        recordingStartTime = Date.now();
        
        // Update UI
        updateRecordingUI(true);
        startRecordingTimer();
        
        showStatusMessage('🎤 Recording started! Speak or play your tabla/percussion sounds.', 'success');
        
    } catch (error) {
        console.error('Error starting recording:', error);
        showStatusMessage('❌ Could not access microphone. Please check permissions.', 'error');
    }
}

// Stop audio recording
function stopRecording() {
    if (mediaRecorder && isRecording) {
        console.log('Stopping recording...');
        
        mediaRecorder.stop();
        isRecording = false;
        
        // Stop all audio tracks
        if (audioStream) {
            audioStream.getTracks().forEach(track => track.stop());
            audioStream = null;
        }
        
        // Update UI
        updateRecordingUI(false);
        stopRecordingTimer();
        
        showStatusMessage('⏹️ Recording stopped. Processing audio...', 'success');
    }
}

// Clear recording
function clearRecording() {
    console.log('Clearing recording...');
    
    recordedAudioBlob = null;
    recordedChunks = [];
    
    // Hide recording info and disable buttons
    const recordingInfo = document.getElementById('recordingInfo');
    const recordingPreview = document.getElementById('recordingPreview');
    
    recordingInfo.style.display = 'none';
    recordingPreview.src = '';
    
    // Reset buttons
    const startBtn = document.getElementById('startRecordBtn');
    const stopBtn = document.getElementById('stopRecordBtn');
    const clearBtn = document.getElementById('clearRecordBtn');
    
    startBtn.disabled = false;
    stopBtn.disabled = true;
    clearBtn.disabled = true;
    
    disableSubmitButtons();
    
    showStatusMessage('🗑️ Recording cleared. Ready to record again.', 'success');
    setTimeout(() => hideStatusMessage(), 2000);
}

// Process recorded audio
async function processRecordedAudio() {
    if (recordedChunks.length === 0) {
        showStatusMessage('❌ No audio data recorded.', 'error');
        return;
    }
    
    console.log(`Processing ${recordedChunks.length} audio chunks...`);
    
    // Create blob from recorded chunks
    const mimeType = recordedChunks[0].type || 'audio/webm';
    const originalBlob = new Blob(recordedChunks, { type: mimeType });
    
    console.log(`Created audio blob: ${originalBlob.size} bytes, type: ${mimeType}`);
    
    try {
        // Convert to WAV if it's WebM
        if (mimeType.includes('webm')) {
            console.log('Converting WebM to WAV for better compatibility...');
            recordedAudioBlob = await convertToWAV(originalBlob);
            console.log(`Converted to WAV: ${recordedAudioBlob.size} bytes`);
        } else {
            recordedAudioBlob = originalBlob;
        }
    } catch (error) {
        console.warn('Audio conversion failed, using original:', error);
        recordedAudioBlob = originalBlob;
    }
    
    // Show recording info and preview
    const recordingInfo = document.getElementById('recordingInfo');
    const recordingPreview = document.getElementById('recordingPreview');
    const recordingStatus = document.getElementById('recordingStatus');
    
    recordingInfo.style.display = 'block';
    
    // Create URL for preview (use original for better browser support)
    const audioURL = URL.createObjectURL(originalBlob);
    recordingPreview.src = audioURL;
    
    const duration = formatDuration(Date.now() - recordingStartTime);
    recordingStatus.textContent = `Recording completed (${duration})`;
    
    // Enable clear button and submit buttons
    document.getElementById('clearRecordBtn').disabled = false;
    enableSubmitButtons();
    
    showStatusMessage('✅ Recording complete! You can now process it or record again.', 'success');
}

// Update recording UI
function updateRecordingUI(recording) {
    const startBtn = document.getElementById('startRecordBtn');
    const stopBtn = document.getElementById('stopRecordBtn');
    const clearBtn = document.getElementById('clearRecordBtn');
    const recordingStatus = document.getElementById('recordingStatus');
    const body = document.body;
    
    if (recording) {
        startBtn.disabled = true;
        stopBtn.disabled = false;
        clearBtn.disabled = true;
        recordingStatus.textContent = 'Recording in progress...';
        body.classList.add('recording-active');
        
        // Show recording info
        document.getElementById('recordingInfo').style.display = 'block';
    } else {
        startBtn.disabled = false;
        stopBtn.disabled = true;
        body.classList.remove('recording-active');
    }
}

// Recording timer
function startRecordingTimer() {
    recordingTimer = setInterval(() => {
        if (recordingStartTime) {
            const elapsed = Date.now() - recordingStartTime;
            const duration = formatDuration(elapsed);
            document.getElementById('recordingDuration').textContent = duration;
        }
    }, 1000);
}

function stopRecordingTimer() {
    if (recordingTimer) {
        clearInterval(recordingTimer);
        recordingTimer = null;
    }
}

// Convert audio blob to WAV format using Web Audio API
async function convertToWAV(audioBlob) {
    try {
        // Create audio context
        const audioContext = new (window.AudioContext || window.webkitAudioContext)();
        
        // Convert blob to array buffer
        const arrayBuffer = await audioBlob.arrayBuffer();
        
        // Decode audio data
        const audioBuffer = await audioContext.decodeAudioData(arrayBuffer);
        
        // Convert to WAV
        const wavBuffer = audioBufferToWav(audioBuffer);
        
        // Create WAV blob
        const wavBlob = new Blob([wavBuffer], { type: 'audio/wav' });
        
        // Close audio context to free resources
        await audioContext.close();
        
        return wavBlob;
    } catch (error) {
        console.error('Audio conversion error:', error);
        throw error;
    }
}

// Convert AudioBuffer to WAV format
function audioBufferToWav(buffer) {
    const length = buffer.length;
    const numberOfChannels = buffer.numberOfChannels;
    const sampleRate = buffer.sampleRate;
    const bitDepth = 16;
    
    const arrayBuffer = new ArrayBuffer(44 + length * numberOfChannels * 2);
    const view = new DataView(arrayBuffer);
    
    // WAV header
    const writeString = (offset, string) => {
        for (let i = 0; i < string.length; i++) {
            view.setUint8(offset + i, string.charCodeAt(i));
        }
    };
    
    writeString(0, 'RIFF');
    view.setUint32(4, 36 + length * numberOfChannels * 2, true);
    writeString(8, 'WAVE');
    writeString(12, 'fmt ');
    view.setUint32(16, 16, true);
    view.setUint16(20, 1, true);
    view.setUint16(22, numberOfChannels, true);
    view.setUint32(24, sampleRate, true);
    view.setUint32(28, sampleRate * numberOfChannels * bitDepth / 8, true);
    view.setUint16(32, numberOfChannels * bitDepth / 8, true);
    view.setUint16(34, bitDepth, true);
    writeString(36, 'data');
    view.setUint32(40, length * numberOfChannels * 2, true);
    
    // Convert audio data to 16-bit PCM
    let offset = 44;
    for (let i = 0; i < length; i++) {
        for (let channel = 0; channel < numberOfChannels; channel++) {
            const sample = Math.max(-1, Math.min(1, buffer.getChannelData(channel)[i]));
            view.setInt16(offset, sample * 0x7FFF, true);
            offset += 2;
        }
    }
    
    return arrayBuffer;
}

// Format duration (milliseconds to MM:SS)
function formatDuration(ms) {
    const seconds = Math.floor(ms / 1000);
    const minutes = Math.floor(seconds / 60);
    const remainingSeconds = seconds % 60;
    return `${minutes.toString().padStart(2, '0')}:${remainingSeconds.toString().padStart(2, '0')}`;
}

// Enable/disable submit buttons
function enableSubmitButtons() {
    submitButtons.forEach(btn => btn.disabled = false);
}

function disableSubmitButtons() {
    submitButtons.forEach(btn => btn.disabled = true);
}

// Initialize the application
document.addEventListener('DOMContentLoaded', function() {
    console.log('File Upload Application with Recording initialized');
    hideStatusMessage();
    hideFileInfo();
    
    // Initialize with upload tab active
    switchInputMethod('upload');
});
