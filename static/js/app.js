const chat = document.getElementById('chat');
const form = document.getElementById('chat-form');
const input = document.getElementById('user-input');
const thinking = document.getElementById('thinking');

if (chat && form && input && thinking) {
  actionButtons();

  form.addEventListener('submit', (e) => {
    e.preventDefault();
    const query = input.value.trim();
    if (!query) return;
    addMessage(query, 'user');
    input.value = '';
    sendQuery(query);
  });
}

function actionButtons() {
  document.querySelectorAll('.suggestion').forEach(btn => {
    btn.addEventListener('click', () => {
      input.value = btn.textContent;
      form.dispatchEvent(new Event('submit'));
    });
  });
}

function addMessage(text, role) {
  const div = document.createElement('div');
  div.className = `message ${role}`;

  const content = document.createElement('div');
  content.className = 'message-content';
  if (role === 'bot' && text) {
    content.innerHTML = formatResponse(text);
  } else {
    content.textContent = text;
  }
  div.appendChild(content);

  const arrow = document.createElement('span');
  arrow.className = 'dialog-arrow';
  div.appendChild(arrow);

  chat.appendChild(div);
  chat.scrollTop = chat.scrollHeight;

  return content;
}

function formatResponse(text) {
  let formatted = text;

  // Convert numbered headings like "1. Title:" to <h4> (remove trailing colon)
  formatted = formatted.replace(/^(\d+\.\s+[^:\n]+):\s*$/gm, '<h4>$1</h4>');

  // Convert standalone headings ending with a colon
  formatted = formatted.replace(/^([A-Za-z][^:\n]+):\s*$/gm, '<h4>$1</h4>');

  // Bold **text** patterns
  formatted = formatted.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');

  // Numbered list items (e.g., "1. item")
  formatted = formatted.replace(/^\s*\d+\.\s+(.+?)$/gm, '<li class="numbered">$1</li>');

  // Bullet points -> list items
  formatted = formatted.replace(/^\s*[-*•]\s+(.+?)$/gm, '<li class="bullet">$1</li>');

  // Wrap consecutive numbered items in <ol>
  formatted = formatted.replace(/(<li class="numbered">.*?<\/li>\s*)+/gs, match => `<ol>${match}</ol>`);

  // Wrap consecutive bullet items in <ul>
  formatted = formatted.replace(/(<li class="bullet">.*?<\/li>\s*)+/gs, match => `<ul>${match}</ul>`);

  // Clean up temporary classes
  formatted = formatted.replace(/<li class="(?:numbered|bullet)">/g, '<li>');

  // Double line breaks -> paragraph breaks
  formatted = formatted.replace(/\n\n+/g, '</p><p>');

  // Single line breaks -> <br>
  formatted = formatted.replace(/\n/g, '<br>');

  // Ensure wrapped in paragraph tags when needed
  if (!formatted.startsWith('<h4>') && !formatted.startsWith('<ul>')) {
    formatted = '<p>' + formatted;
  }
  if (!formatted.endsWith('</p>') && !formatted.endsWith('</ul>')) {
    formatted = formatted + '</p>';
  }

  return formatted;
}

async function sendQuery(query) {
  thinking.classList.remove('hidden');
  const content = addMessage('', 'bot');
  content.innerHTML = '<span class="typing-cursor"></span>';

  const response = await fetch('/api/query', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ query })
  });

  const reader = response.body.getReader();
  const decoder = new TextDecoder('utf-8');
  let botText = '';

  while (true) {
    const { value, done } = await reader.read();
    if (done) break;
    const chunk = decoder.decode(value, { stream: true });
    chunk.split('\n').forEach(line => {
      if (line.startsWith('data:')) {
        const data = line.replace('data:','').trim();
        if (data === '[DONE]') return;
        try {
          const obj = JSON.parse(data);
          botText += obj.token;
          content.innerHTML = formatResponse(botText) + '<span class="typing-cursor"></span>';
          chat.scrollTop = chat.scrollHeight;
        } catch {}
      }
    });
  }
  content.innerHTML = formatResponse(botText);
  thinking.classList.add('hidden');
}
