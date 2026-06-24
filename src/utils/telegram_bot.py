import os
from src.main       import build_graph
from telegram       import Update
from telegram.ext   import Application, CommandHandler, MessageHandler, filters, ContextTypes


# Initialize the LangGraph application once globally
graph_app = build_graph()


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Send a welcome message when the command /start is issued."""
    welcome_message = (
        "Hello! I am your Personal Research Agent 🤖\n\n"
        "Send me a topic you want to research (e.g., 'Find news about AI'), "
        "and I will scrape, analyze, and summarize the best articles for you."
    )
    await update.message.reply_text(welcome_message)


async def handle_message(update: Update, context: ContextTypes.DEFAULT_TYPE) -> None:
    """Process user messages using the LangGraph workflow."""
    user_message = update.message.text
    
    # 1. Send a waiting message so the user knows the bot is working
    waiting_msg = await update.message.reply_text("⏳ Thinking and researching... This might take a minute.")
    
    # 2. Setup initial state for LangGraph
    initial_state = {
        "user_input": user_message,
        "intent": "",
        "user_preferences": {},
        "urls": [],
        "filtered_articles": [],
        "analyzed_reports": [],
        "final_digest": ""
    }
    
    # 3. Run the workflow
    try:
        # We use invoke() to run the graph from start to finish
        final_state = graph_app.invoke(initial_state)
        
        digest = final_state.get("final_digest", "Could not generate a digest.")
        
        # 4. Edit the waiting message with the final result
        # Note: We don't use parse_mode="Markdown" here to avoid formatting errors 
        # if the LLM generates unsupported markdown characters.
        await waiting_msg.edit_text(digest)
        
    except Exception as e:
        print(f"--> [Telegram Bot] Error executing graph: {e}")
        await waiting_msg.edit_text("❌ An error occurred while researching. Please try again.")

def main() -> None:
    """Start the bot."""
    token = os.environ.get("TELEGRAM_BOT_TOKEN")
    if not token:
        print("Error: TELEGRAM_BOT_TOKEN not found in environment variables.")
        return

    # Create the Application and pass it your bot's token
    application = Application.builder().token(token).build()

    # Register handlers
    application.add_handler(CommandHandler("start", start))
    application.add_handler(MessageHandler(filters.TEXT & ~filters.COMMAND, handle_message))

    # Run the bot until the user presses Ctrl-C
    print("--> [Telegram Bot] Bot is running and listening for messages...")
    application.run_polling(allowed_updates=Update.ALL)

if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()
    main()
