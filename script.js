import { MediaRecorder, register } from 'https://jspm.dev/extendable-media-recorder';
import { connect } from 'https://jspm.dev/extendable-media-recorder-wav-encoder';

let mediaRecorder;
let recordedChunks = [];
const recordButton = document.getElementById('recordButton');
const audioInput = document.getElementById('audio');
const form = document.getElementById('evalForm');
const resultDiv = document.getElementById('result');
const audioPlayer = document.getElementById('audioPlayer');

// Initialize extendable-media-recorder for WAV
async function setupRecorder() {
    try {
        const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
        await register(await connect());
        mediaRecorder = new MediaRecorder(stream, { mimeType: 'audio/wav' });

        mediaRecorder.ondataavailable = (e) => {
            if (e.data.size > 0) recordedChunks.push(e.data);
        };

        mediaRecorder.onstop = () => {
            const blob = new Blob(recordedChunks, { type: 'audio/wav' });
            const file = new File([blob], 'recording.wav', { type: 'audio/wav' });
            const dataTransfer = new DataTransfer();
            dataTransfer.items.add(file);
            audioInput.files = dataTransfer.files;
            audioPlayer.src = URL.createObjectURL(blob);
            audioPlayer.style.display = 'block';
            recordButton.textContent = 'Start Recording';
            recordButton.classList.remove('recording');
        };
    } catch (err) {
        resultDiv.innerHTML = `<p style="color: red;">Error accessing microphone: ${err.message}</p>`;
    }
}
setupRecorder()

recordButton.addEventListener('click', async () => {
    if (!mediaRecorder) {
        await setupRecorder();
        return;
    }
    if (mediaRecorder.state === 'recording') {
        mediaRecorder.stop();
    } else {
        recordedChunks = [];
        mediaRecorder.start();
        recordButton.textContent = 'Stop Recording';
        recordButton.classList.add('recording');
    }
});

function formatScore(score) {
    return (Math.round(score * 10) / 10).toString();
}

form.addEventListener('submit', async (e) => {
    e.preventDefault();
    if (!audioInput.files || audioInput.files.length === 0) {
        resultDiv.innerHTML = '<p style="color: red;">Please upload or record an audio file.</p>';
        return;
    }

    resultDiv.innerHTML = 'Processing...';
    const formData = new FormData(form);

    try {
        const response = await fetch('http://localhost:5000/evaluate', {
            method: 'POST',
            body: formData
        });

        if (!response.ok) {
            const errorData = await response.json();
            throw new Error(errorData.error || 'Unknown error');
        }

        const data = await response.json();
        console.log('Received json:', data);
        /*resultDiv.innerHTML = `
            <h2>GOPMaxLogit Results</h2>
            <pre id="json-output">${JSON.stringify(data, null, 2)}</pre>

        `;*/

        resultDiv.innerHTML = `
            <h2>GOPMaxLogit Analysis</h2>
                <p>Model Output: ${data.model_output}</p>
            <h3>Phoneme Details</h3>
            <div class="list">
                ${data.phonemes.map(phoneme => `
                    <div class="item ${phoneme.confidence}">
                        <div class="item-header">
                            <span class="score">${formatScore(phoneme.gop_maxlogit)}</span>
                            <span class="label">${phoneme.jyutping}</span>
                        </div>
                        <div class="item-details">
                            <p>Other Predictions:</p>
                            <ol>
                                ${phoneme.prediction.map(pred => `<li>${pred[0]}: ${formatScore(pred[1])}</li>`).join('')}
                            </ol>
                        </div>
                    </div>
                `).join('')}
            </div>
            <div>
                <h3>Tone Details</h3>
                <div class="list">
                    ${data.tones.map(tone => `
                        <div class="item ${tone.confidence}">
                            <div class="item-header">
                                <span class="score">${formatScore(tone.gop_maxlogit)}</span>
                                <span class="label">${tone.tone}</span>
                            </div>
                            <div class="item-details">
                                <p>Other Predictions:</p>
                                <ol>
                                    ${tone.prediction.map(pred => `<li>${pred[0]}: ${formatScore(pred[1])}</li>`).join('')}
                                </ol>
                            </div>
                        </div>
                    `).join('')}
                </div>
            </div>
            <details>
                <summary>Raw JSON</summary>
                <pre>${JSON.stringify(data, null, 2)}</pre>
            </details>
        `;

    } catch (error) {
        resultDiv.innerHTML = `<p style="color: red;">Error: ${error.message}</p>`;
    }
});