export class SSEClient {
  private reader: ReadableStreamDefaultReader<Uint8Array> | null = null;
  private decoder = new TextDecoder();
  private buffer = '';

  async connect(
    url: string,
    body: any,
    onMessage: (data: any) => void,
    onError?: (error: any) => void
  ) {
    try {
      const response = await fetch(url, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify(body),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No reader available');
      }

      this.reader = reader;
      this.buffer = '';

      // Read the stream
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        const text = this.decoder.decode(value, { stream: true });
        this.buffer += text;
        
        // Process complete messages (ending with \n\n)
        const messages = this.buffer.split('\n\n');
        
        // Keep the last (potentially incomplete) message in the buffer
        this.buffer = messages[messages.length - 1];
        
        // Process all complete messages
        for (let i = 0; i < messages.length - 1; i++) {
          const message = messages[i];
          const lines = message.split('\n');
          
          for (const line of lines) {
            if (line.startsWith('data: ')) {
              const data = line.slice(6);
              if (data.trim()) {
                try {
                  const parsed = JSON.parse(data);
                  onMessage(parsed);
                } catch (e) {
                  console.error('Failed to parse SSE data:', e, data);
                }
              }
            }
          }
        }
      }
    } catch (error) {
      if (onError) {
        onError(error);
      } else {
        console.error('SSE connection error:', error);
      }
    }
  }

  close() {
    if (this.reader) {
      this.reader.cancel();
      this.reader = null;
    }
  }
}