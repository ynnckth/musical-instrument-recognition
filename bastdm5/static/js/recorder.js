((() => {
    const AudioContext = window.AudioContext || window.webkitAudioContext;
    const audioContext = new AudioContext;

    const recordButton = document.getElementById('record-button');

    let recorder;
    let micStream;
    let isRecording = false;

    const startRecording = async () => {
        isRecording = true;
        try {
            const stream = await navigator.mediaDevices.getUserMedia({audio: true, video: false});
            micStream = stream;
            const input = audioContext.createMediaStreamSource(stream);
            recorder = new Recorder(input, {numChannels: 1});
            recorder.record();
            console.log("Started recording ...");
        } catch (error) {
            isRecording = false;
            console.log('Error starting recording', error);
        }
    };

    const stopRecording = () => {
        recorder.stop();
        micStream.getAudioTracks()[0].stop();
        isRecording = false;
        console.log('Stopped recording');
    };

    const uploadRecording = async (blob) => {
        const formData = new FormData();
        formData.append('file', blob, 'recording.wav');

        const response = await fetch('/upload', {
            method: 'POST',
            body: formData
        });

        const res = await response.text();

        document.open();
        document.write(res);
        document.close();
    };

    const toggleRecording = async () => {
        if (isRecording === false) {
            await startRecording();
        } else {
            stopRecording();
            recorder.exportWAV(uploadRecording);
        }
    };

    recordButton.addEventListener('click', toggleRecording);
})());

