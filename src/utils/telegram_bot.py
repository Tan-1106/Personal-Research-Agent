import os
import pytz
import datetime
from src.main       import build_graph
from telegram       import Update
from telegram.ext   import Application, CommandHandler, MessageHandler, filters, ContextTypes


# Initialize the LangGraph application once globally
graph_app = build_graph()


def split_message(text: str, max_length: int = 4000) -> list[str]:
    """Splits a long message into chunks, prioritizing paragraph breaks to maintain context."""
    if len(text) <= max_length:
        return [text]
        
    chunks = []
    while text:
        if len(text) <= max_length:
            chunks.append(text)
            break
            
        # 1. Find the nearest double newline (paragraph break) within the max limit
        split_at = text.rfind('\n\n', 0, max_length)
        if split_at == -1:
            # 2. If no paragraph break is found, fallback to a single newline
            split_at = text.rfind('\n', 0, max_length)
            if split_at == -1:
                # 3. If no newline is found, fallback to the nearest space
                split_at = text.rfind(' ', 0, max_length)
                if split_at == -1:
                    # 4. As a last resort, hard split at exactly the max_length
                    split_at = max_length
                    
        chunks.append(text[:split_at].strip())
        text = text[split_at:].strip()
        
    return chunks


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message and print the user's Chat ID."""
    chat_id = update.message.chat_id
    welcome_message = (
        "Hello! I am your Personal Research Agent 🤖\n\n"
        f"🔑 **Your Chat ID is:** `{chat_id}`\n\n"
        "Please add this ID to your `.env` file as `TELEGRAM_ADMIN_ID` to enable morning summaries.\n"
        "Send me any topic to start researching!"
    )
    await update.message.reply_text(welcome_message, parse_mode="Markdown")


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process user messages using the LangGraph workflow."""
    user_message = update.message.text
    chat_id = str(update.message.chat_id)
    config = {"configurable": {"thread_id": chat_id}}
    
    # Extract current memory to get the dynamically translated system messages
    state_snapshot = graph_app.get_state(config)
    current_state = state_snapshot.values if state_snapshot else {}
    user_prefs = current_state.get("user_preferences", {})
    
    # Default English fallback
    sys_msgs = user_prefs.get("system_messages", {
        "thinking": "⏳ Thinking and researching... This might take a minute.",
        "error": "❌ An error occurred while researching. Please try again.",
        "morning": "🌅 Good morning! Preparing your daily digest..."
    })
    
    # 1. Send the dynamically translated waiting message
    waiting_msg = await update.message.reply_text(sys_msgs["thinking"])
    
    # 2. Setup inputs and run graph
    inputs = {"user_input": user_message}
    
    try:
        final_state = graph_app.invoke(inputs, config=config)
        digest = final_state.get("final_digest", "Could not generate a digest.")
        
        # Split the long message into chunks
        chunks = split_message(digest)
        
        # Replace the "Thinking..." message with the first chunk
        await waiting_msg.edit_text(chunks[0])
        
        # Send any remaining chunks as sequential new messages
        for chunk in chunks[1:]:
            await context.bot.send_message(chat_id=chat_id, text=chunk)
        
    except Exception as e:
        print(f"--> [Telegram Bot] Error executing graph: {e}")
        await waiting_msg.edit_text(sys_msgs["error"])
        
        
async def scheduled_daily_research(context: ContextTypes.DEFAULT_TYPE) -> None:
    """Runs automatically at 7:00 AM to fetch daily news."""
    chat_id = os.environ.get("TELEGRAM_ADMIN_ID")
    if not chat_id:
        print("--> [Cron] TELEGRAM_ADMIN_ID not set. Skipping daily research.")
        return
        
    # Fetch memory to say Good Morning in the correct language
    config = {"configurable": {"thread_id": chat_id}}
    state_snapshot = graph_app.get_state(config)
    current_state = state_snapshot.values if state_snapshot else {}
    sys_msgs = current_state.get("user_preferences", {}).get("system_messages", {
        "morning": "🌅 Good morning! Preparing your daily digest based on your preferences...",
        "error": "❌ An error occurred during the morning routine."
    })
    print("--> [Cron] Starting scheduled daily research...")
    await context.bot.send_message(chat_id=chat_id, text=sys_msgs["morning"])
    
    inputs = {"user_input": "Find the most important updates and news from the last 24 hours."}
    
    try:
        final_state = graph_app.invoke(inputs, config=config)
        digest = final_state.get("final_digest", "Could not generate digest.")
        
        chunks = split_message(digest)
        for chunk in chunks:
            await context.bot.send_message(chat_id=chat_id, text=chunk)

    except Exception as e:
        print(f"--> [Cron] Error: {e}")
        await context.bot.send_message(chat_id=chat_id, text=sys_msgs.get("error", "❌ An error occurred during the morning routine."))


def main() -> None:
    """Start the bot."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment variables.")
        return

    # Create the Application and pass it your bot's token
    application = Application.builder().token(token).build()
    
    # Configure Timezone
    tz_string = os.environ.get("TIMEZONE", "UTC")
    local_tz = pytz.timezone(tz_string)
    
    # Schedule the daily job at 7:00 AM using the defined Timezone
    target_time = datetime.time(hour=7, minute=0, second=0, tzinfo=local_tz)
    application.job_queue.run_daily(scheduled_daily_research, time=target_time)

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    print("--> [Telegram Bot] Bot is running and listening for messages...")
    application.run_polling(allowed_updates=Update.ALL_TYPES)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
