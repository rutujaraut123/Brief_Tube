document.getElementById('get-transcript').addEventListener('click', function() {
    const videoUrl = document.getElementById('video-url').value;
    
    // Extract the video ID from the URL
    let videoId = null;

    try {
        const urlParams = new URL(videoUrl).searchParams;
        videoId = urlParams.get('v');

        if (!videoId && videoUrl.includes('youtu.be')) {
            videoId = videoUrl.split('youtu.be/')[1];
        }

        if (!videoId) {
            throw new Error('Invalid YouTube URL. Please enter a valid URL.');
        }

    } catch (error) {
        document.getElementById('transcript-output').innerText = error.message;
        return;
    }

    fetch('http://127.0.0.1:5000/get_transcript', {
        method: 'POST',
        headers: {
            'Content-Type': 'application/json',
        },
        body: JSON.stringify({ video_id: videoUrl })  // Pass the full video URL to the backend
    })
    .then(response => response.json())
    .then(data => {
        const outputDiv = document.getElementById('transcript-output');
        outputDiv.innerHTML = '';

        if (data.error) {
            outputDiv.innerText = 'Error: ' + data.error;  // Handle errors from the backend
        } else {
            // Display the transcript directly if returned as plain text
            outputDiv.innerHTML = `<p>${data.transcript}</p>`;
        }
    })
    .catch(error => console.error('Error:', error));
});
