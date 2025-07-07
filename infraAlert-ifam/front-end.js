
function fazerLogin() {
    const usuario = document.getElementById('usuario').value;
    const senha = document.getElementById('senha').value;

    fetch('http://localhost:5500/login', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        credentials: 'include',
        body: JSON.stringify({ usuario, senha })
    })
    .then(res => {
        if (res.ok) return res.json();
        throw new Error('Credenciais inválidas');
    })
    .then(data => {
        document.getElementById('loginContainer').style.display = 'none';
        document.getElementById('sistema').style.display = 'block';
    })
    .catch(err => {
        document.getElementById('erroLogin').innerText = err.message;
        console.log(`Error: ${err.mensag}`)
    });
}

const mockData = [
    { date: '2025-06-24', risk: false },
    { date: '2025-06-25', risk: true },
    { date: '2025-06-26', risk: false },
    { date: '2025-06-27', risk: true },
    { date: '2025-06-28', risk: false }
];

const ctx = document.getElementById('chart').getContext('2d');
new Chart(ctx, {
    type: 'bar',
    data: {
    labels: mockData.map(d => d.date),
    datasets: [{
        label: 'Detecções de Rachadura (críticas em vermelho)',
        data: mockData.map(d => d.risk ? 2 : 1),
        backgroundColor: mockData.map(d => d.risk ? '#D32F2F' : '#1976D2'),
    }]
    },
    options: {
    scales: {
        y: {
        beginAtZero: true,
        ticks: {
            stepSize: 1,
            callback: value => value === 2 ? 'Crítica' : 'Normal'
        }
        }
    }
    }
});

const historyList = document.getElementById('history');
mockData.forEach(item => {
    const li = document.createElement('li');
    li.textContent = `${item.date} - ${item.risk ? 'Rachadura Crítica' : 'Rachadura Normal'}`;
    if (item.risk) li.style.color = '#D32F2F';
    historyList.appendChild(li);
});

if (mockData.some(d => d.risk)) {
    document.getElementById('alertBox').style.display = 'block';
}

let imagemSelecionada;
document.getElementById('imageUpload').addEventListener('change', function(event) {
    imagemSelecionada = event.target.files[0];
    const reader = new FileReader();
    reader.onload = function(e) {
    document.getElementById('preview').src = e.target.result;
    };
    reader.readAsDataURL(imagemSelecionada);
});

let stream;
async function startCamera() {
    const video = document.getElementById('camera');
    const captureBtn = document.getElementById('captureBtn');
    stream = await navigator.mediaDevices.getUserMedia({ video: true });
    video.srcObject = stream;
    video.style.display = 'block';
    captureBtn.style.display = 'block';
}

function capturePhoto() {
    const video = document.getElementById('camera');
    const canvas = document.getElementById('snapshotCanvas');
    const context = canvas.getContext('2d');
    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;
    context.drawImage(video, 0, 0);
    canvas.toBlob(blob => {
    imagemSelecionada = new File([blob], 'foto.png', { type: 'image/png' });
    document.getElementById('preview').src = URL.createObjectURL(blob);
    }, 'image/png');
    video.style.display = 'none';
    document.getElementById('captureBtn').style.display = 'none';
    stream.getTracks().forEach(track => track.stop());
}

async function enviarImagem() {
    if (!imagemSelecionada) {
    alert("Nenhuma imagem selecionada.");
    return;
    }
    const formData = new FormData();
    formData.append('imagem', imagemSelecionada);

    try {
    const resposta = await fetch('http://localhost:5000/detectar', {
        method: 'POST',
        body: formData,
        credentials: 'include'
    });

    const resultado = await resposta.json();
    document.getElementById('resultadoTexto').textContent = resultado.mensagem;
    if (resultado.risco) {
        document.getElementById('alertBox').style.display = 'block';
    }

    const imagemResp = await fetch('http://localhost:5000/resultado');
    const blob = await imagemResp.blob();
    document.getElementById('imagemProcessada').src = URL.createObjectURL(blob);
    document.getElementById('imagemProcessada').style.display = 'block';

    } catch (error) {
    console.error("Erro ao enviar imagem:", error);
    }
}
