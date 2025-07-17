import re
from pathlib import Path

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 100):
    """Same chunking logic as in your pipeline"""
    sentences = re.split(r'(?<=[.!?])\s+', text)
    chunks = []
    current_chunk = []
    current_length = 0
    
    for sentence in sentences:
        sent_length = len(sentence)
        if current_length + sent_length > chunk_size and current_chunk:
            chunks.append(" ".join(current_chunk))
            overlap_count = max(1, int(len(current_chunk) * 0.3))
            current_chunk = current_chunk[-overlap_count:]
            current_length = sum(len(s) for s in current_chunk)
        
        current_chunk.append(sentence)
        current_length += sent_length
    
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    
    return chunks

def debug_chunks():
    """Debug function to see how documents are being chunked"""
    
    # Read the I2Connect document
    file_path = Path("documents/I2Connect_Traffic_Safety_System.md")
    
    if not file_path.exists():
        print(f"❌ File not found: {file_path}")
        return
    
    with open(file_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    print(f"📄 Document length: {len(content)} characters")
    
    chunks = chunk_text(content, chunk_size=500, overlap=100)
    
    print(f"📦 Total chunks: {len(chunks)}")
    print("\n" + "="*80)
    
    # Look for chunks containing partner information
    partner_keywords = ['university', 'scania', 'smart eye', 'viscando', 'partners', 'consortium', 'collaborators']
    
    partner_chunks = []
    for i, chunk in enumerate(chunks):
        chunk_lower = chunk.lower()
        if any(keyword in chunk_lower for keyword in partner_keywords):
            partner_chunks.append((i, chunk))
    
    print(f"🔍 Found {len(partner_chunks)} chunks with partner keywords")
    
    for i, chunk in partner_chunks:
        print(f"\n📋 CHUNK {i} (contains partner info):")
        print("-" * 60)
        print(chunk[:500] + "..." if len(chunk) > 500 else chunk)
        print("-" * 60)
    
    # Also show the first few chunks to see general structure
    print(f"\n\n📄 FIRST 3 CHUNKS FOR REFERENCE:")
    for i in range(min(3, len(chunks))):
        print(f"\n📋 CHUNK {i}:")
        print("-" * 40)
        print(chunks[i][:300] + "..." if len(chunks[i]) > 300 else chunks[i])
        print("-" * 40)
    
    # Look for the collaborators section specifically
    if "collaborators" in content.lower():
        print("\n🔍 SEARCHING FOR COLLABORATORS SECTION:")
        # Find the section around "collaborators"
        collab_pos = content.lower().find("collaborators")
        if collab_pos != -1:
            start = max(0, collab_pos - 200)
            end = min(len(content), collab_pos + 800)
            section = content[start:end]
            print("-" * 60)
            print(section)
            print("-" * 60)

if __name__ == "__main__":
    debug_chunks()