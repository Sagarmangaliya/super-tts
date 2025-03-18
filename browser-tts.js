import { pipeline } from '@xenova/transformers';

class BrowserTTS {
  static instance = null;

  static async getInstance() {
    if (!this.instance) {
      this.instance = await pipeline('text-to-speech', 'Xenova/speecht5_tts');
    }
    return this.instance;
  }

  async speak(text, speaker_embeddings) {
    const tts = await BrowserTTS.getInstance();
    return tts(text, { speaker_embeddings });
  }
}
