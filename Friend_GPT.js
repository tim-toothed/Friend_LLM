import { HfInference } from '@huggingface/inference';

const state = { pendingPrompts: {} };

export default {
  async fetch(request, env) {
    const hf = new HfInference(env.HUGGING_FACE_API_KEY);

    if (request.method === "POST") {
      const payload = await request.json().catch(() => new Response('Invalid JSON', { status: 400 }));
      const { message } = payload;
      if (message) {
        const { chat: { id: chatId }, from: { id: userId, username }, text: input } = message;
        const commands = { "/stas_gpt": this.initiateStasGpt };
        const command = Object.keys(commands).find(cmd => input.startsWith(cmd));

        if (command) {
          const allowedUsers = {
            ids: env.ALLOWED_IDS.split(", ").map(Number),
            usernames: env.ALLOWED_USERNAMES.split(", "),
            chatIds: env.ALLOWED_CHAT_IDS.split(", ").map(Number)
          };
          if (!allowedUsers.ids.includes(userId) && !allowedUsers.usernames.includes(username) && !allowedUsers.chatIds.includes(chatId)) {
            await this.sendMessageHTML(env.API_KEY, chatId, "<b>❌This function is not available to you</b>");
            return new Response('OK');
          }
          await this.callHandler(commands[command], env.API_KEY, chatId, userId, hf, input);
        } else if (state.pendingPrompts[userId]) {
          const { promptType, modelName } = state.pendingPrompts[userId];
          await this.handleParamsGpt(hf, env.API_KEY, chatId, input, userId, promptType, modelName, state);
        }
      }
    }
    return new Response('OK');
  },

  async callHandler(handler, apiKey, chatId, userId, hf, text) {
    await handler.call(this, apiKey, chatId, userId, 'standard', 'TOOTHED/stas-gpt-new');
  },

  async initiateStasGpt(apiKey, chatId, userId, promptType, modelName) {
    state.pendingPrompts[userId] = { promptType, modelName };
    const promptMessage = await this.sendMessageHTML(apiKey, chatId, '<b>💬Enter prompt for Stas GPT:</b>', "HTML", {
      inline_keyboard: [[{ text: "Cancel", callback_data: "cancel_prompt" }]]
    });
    state.pendingPrompts[userId].messageId = promptMessage.result.message_id;
  },

  async handleParamsGpt(hf, apiKey, chatId, prompt, userId, promptType, modelName, state) {
    const promptMessageId = state.pendingPrompts[userId].messageId;
    if (promptMessageId) await this.deleteMessage(apiKey, chatId, promptMessageId);

    const loadingMessage = await this.sendMessageHTML(apiKey, chatId, `⏳Loading...\n\n❗️If the response does not arrive within 3 minutes, it means an error has occurred. Please try again.`);
    const loadingMessageId = loadingMessage.result.message_id;

    const parameters = {
      max_new_tokens: promptType === 'large' ? 400 : 250,
      num_return_sequences: 1,
      no_repeat_ngram_size: 3,
      do_sample: true,
      top_k: promptType === 'large' ? 50 : 100,
      top_p: promptType === 'large' ? 0.9 : 0.95,
      temperature: promptType === 'large' ? 0.7 : 0.9,
      num_beams: promptType === 'large' ? 9 : undefined,
      repetition_penalty: promptType === 'large' ? 2.0 : 1.1,
      early_stopping: true,
    };

    const generated_texts = await this.generateTextFromHuggingFace(hf, prompt, parameters, apiKey, chatId, userId, modelName);
    if (generated_texts) {
      await this.deleteMessage(apiKey, chatId, loadingMessageId);
      await this.sendMessageHTML(apiKey, chatId, generated_texts);
      delete state.pendingPrompts[userId];
    }
  },

  async generateTextFromHuggingFace(hf, prompt, parameters, apiKey, chatId, userId, modelName) {
    const timeout = 500 * 10000;
    const timer = new Promise((_, reject) => setTimeout(() => reject(new Error("Timeout reached")), timeout));

    try {
      const response = await Promise.race([
        hf.textGeneration({
          accessToken: hf.accessToken,
          model: modelName,
          inputs: prompt,
          parameters: parameters,
          options: { wait_for_model: true }
        }),
        timer
      ]);

      if (!response) {
        await this.sendMessageHTML(apiKey, chatId, "Model has crashed");
        throw new Error("Timeout reached");
      }

      if (response.generated_text) {
        delete state.pendingPrompts[userId];
        return [response.generated_text];
      } else {
        console.error(`Unexpected response format from Hugging Face API: ${JSON.stringify(response)}`);
        delete state.pendingPrompts[userId];
        return [`Unexpected response format from Hugging Face API`];
      }
    } catch (error) {
      console.error(`Error fetching from Hugging Face API: ${error.message}`);
      delete state.pendingPrompts[userId];
      return ["Model has crashed"];
    }
  },

  async deleteMessage(apiKey, chatId, messageId) {
    const url = `https://api.telegram.org/bot${apiKey}/deleteMessage`;
    const payload = { chat_id: chatId, message_id: messageId };
    try {
      await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
    } catch (error) {
      console.error(`Error deleting message: ${error}`);
    }
  },

  async sendMessageHTML(apiKey, chatId, text, parseMode = "HTML", replyMarkup = null) {
    const url = `https://api.telegram.org/bot${apiKey}/sendMessage`;
    const payload = { chat_id: chatId, text: Array.isArray(text) ? text[0] : text, parse_mode: parseMode, reply_markup: replyMarkup ? JSON.stringify(replyMarkup) : undefined };
    try {
      const response = await fetch(url, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify(payload) });
      return await response.json();
    } catch (error) {
      console.error(`Error sending message: ${error}`);
      return Response('Error sending message', { status: 500 });
    }
  }
};