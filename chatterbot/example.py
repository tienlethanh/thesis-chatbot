from chatterbot import ChatBot

bot = ChatBot(
    "Chatterbot",
    logic_adapters=[
        'chatterbot_item.item_adapter.MyLogicAdapter'
    ],
    input_adapter="chatterbot.input.TerminalAdapter",
    output_adapter="chatterbot.output.TerminalAdapter",
)

print("Type something to begin...")

while True:
    try:

        bot_input = bot.get_response(None)

    # Press ctrl-c or ctrl-d on the keyboard to exit
    except (KeyboardInterrupt, EOFError, SystemExit):
        break
