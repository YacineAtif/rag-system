import re
from pathlib import Path

def chunk_text(text: str, chunk_size: int = 2000, overlap: int = 200):
    """Chunk text using large segments while preserving bullet lists."""

    bullet_re = re.compile(r"^\s*(?:[-*]|\d+\.)\s+")
    lines = text.splitlines()
    segments = []
    buffer = []
    in_bullet = False

    for line in lines:
        if bullet_re.match(line):
            if not in_bullet and buffer:
                segments.append(" ".join(buffer).strip())
                buffer = []
            in_bullet = True
            buffer.append(line.strip())
        else:
            if in_bullet and buffer:
                segments.append("\n".join(buffer).strip())
                buffer = []
            in_bullet = False
            buffer.append(line.strip())

    if buffer:
        if in_bullet:
            segments.append("\n".join(buffer).strip())
        else:
            segments.append(" ".join(buffer).strip())

    chunks = []
    current_chunk = []
    current_tokens = 0

    for seg in segments:
        seg_tokens = len(seg.split())
        if current_tokens + seg_tokens > chunk_size and current_chunk:
            chunks.append("\n".join(current_chunk))
            if overlap > 0:
                tokens = "\n".join(current_chunk).split()
                overlap_tokens = tokens[-overlap:]
                current_chunk = [" ".join(overlap_tokens)]
                current_tokens = len(overlap_tokens)
            else:
                current_chunk = []
                current_tokens = 0
        current_chunk.append(seg)
        current_tokens += seg_tokens

    if current_chunk:
        chunks.append("\n".join(current_chunk))

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
    
    chunks = chunk_text(content, chunk_size=2000, overlap=200)
    
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