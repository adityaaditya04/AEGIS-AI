const API_BASE = window.API_BASE || 'http://127.0.0.1:8000';

function appendMessage(text, cls='bot'){
  const messages = document.getElementById('messages');
  const el = document.createElement('div');
  el.className = `message ${cls}`;
  el.textContent = text;
  messages.appendChild(el);
  messages.scrollTop = messages.scrollHeight;
}

async function classifyPrompt(text){
  const res = await fetch(`${API_BASE}/classify`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  return res.json();
}

async function proxyPrompt(text){
  const res = await fetch(`${API_BASE}/proxy`, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ text })
  });
  return res.json();
}

document.getElementById('promptForm').addEventListener('submit', async (e) =>{
  e.preventDefault();
  const input = document.getElementById('promptInput');
  const text = input.value.trim();
  if(!text) return;

  appendMessage(text, 'user');
  input.value = '';

  appendMessage('Checking prompt for injection...', 'bot');
  try{
    const cl = await classifyPrompt(text);
    // remove the checking message
    const messages = document.getElementById('messages');
    if(messages.lastChild && messages.lastChild.textContent.includes('Checking prompt')){
      messages.removeChild(messages.lastChild);
    }

    if(cl.blocked){
      appendMessage(`Blocked: prompt classified as malicious (score=${cl.score})`, 'bot');
      return;
    }

    appendMessage('Prompt is safe — forwarding to protected LLM...', 'bot');
    const reply = await proxyPrompt(text);
    appendMessage(reply.text || '[no reply]', 'bot');
  }catch(err){
    appendMessage('Error contacting server: ' + err.message, 'bot');
  }
});
