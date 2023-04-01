import discord
from discord.ext import commands
from discord.ext.commands import Bot
import praw
import os
import nltk
nltk.download('punkt')
nltk.download('wordnet')
import time
import webbrowser
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np
from keras.models import load_model
model = load_model('tadpolemodel.h5')
import json
import random
import imdb
import wikipedia

#chatbot conversation section

convos = json.loads(open('convo.json').read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence
def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s:
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"convo": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, convo_json, msg):
    if float(ints[0]['probability']) > 0.98:
        tag = ints[0]['convo']
        list_of_convos = convo_json['convos']
        for i in list_of_convos:
            if(i['tag']== tag):
                result = random.choice(i['responses'])
                break
    else:
        #result = "let me google that"
        url = "https://google.com/search?q="
        newlink = ""
        for word in msg.split()[1:]:
            newlink = "+".join((newlink, word))
        result = "".join((url, newlink))

    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, convos, msg)
    return res

#discord activity section: a discord client is created to interact with the user

client = commands.Bot(command_prefix = '!', help_command = None)

#reddit browsing section: a reddit api is used to scapre data from given subreddits
reddit = praw.Reddit(client_id = "REDDIT CLIENT ID",
                    client_secret = "REDDIT CLIENT SECRET",
                    username = "REDDIT USERNAME",
                    password = "REDDIT PASSWORD",
                    user_agent = "tadpole")
#client_id and client_secret must not be shared with anyone, you should create your own.

#commands executed first when the bot goes online
@client.event
async def on_ready():
    print('We have logged in as {0.user}'.format(client))
    await client.change_presence(activity=discord.Activity(type=discord.ActivityType.watching, name="you 🤖 // !help"))

#The first message Tadpole sends when invited to a server
@client.event
async def on_guild_join(guild):
    for channel in guild.text_channels:
        if channel.permissions_for(guild.me).send_messages:
            title = "Hello, there!"
            description = "I am Tadpole, and I've just been invited to join this server. Type **!help** to understand more about what I can do 🦾"
            emb = discord.Embed(title = title, description = description, color = 0xf4fc58)
            await channel.send(embed = emb)
        break

#constantly runs, takes messages from the user and responds using the provided dataset
@client.event
async def on_message(message):
    msg = message.content.strip()
    if message.author == client.user:
        return

    if message.content.startswith(msg) and client.user.mentioned_in(message) and not message.content.startswith('!'):
        if "https://google.com/search?q=" not in chatbot_response(msg):
            await message.channel.send(chatbot_response(msg))
        else:
            link = chatbot_response(msg)
            #emb = discord.Embed(description = f"**[This]({link})** " + msg.split(" ", 1)[1], color = 0xf4fc58)
            emb = discord.Embed(description = "Oof. I may be smart but I can't answer that yet. However, I found " + f"**[This]({link})** " + "on Google for you question", color = 0xf4fc58)
            await message.channel.send(embed = emb)
    await client.process_commands(message)

#8ball pool command, runs using: !8ball <enter question>
@client.command(aliases=['8ball'])
async def _8ball(ctx, *, message: str):
    reply = ["It is certain", "It is decidedly so", "Without a doubt", "Yes, definitely",
               "You may rely on it", "As I see it, yes", "Most Likely", "Outlook Good",
               "Yes", "Signs point to yes", "Reply hazy, try again", "Ask again later",
               "Better not tell you now", "Cannot predict now", "Concentrate and ask again",
               "Don't count on it", "My reply is no", "My sources say no", "Outlook not so good", "Very Doubtful"]
    await ctx.send(f'Question: {message}\nAnswer: {random.choice(reply)}')

#meme command, runs using !meme
@client.command()
async def meme(ctx):
    subreddit = reddit.subreddit("memes")
    meme_list = []
    top = subreddit.top(limit = 25)
    for submission in top:
        meme_list.append(submission)
    random_meme = random.choice(meme_list)

    name = random_meme.title
    url = random_meme.url

    emb = discord.Embed(title = name, color = 0xf4fc58)
    emb.set_image(url = url)
    await ctx.send(embed = emb)

#joke command, runs using !joke
@client.command()
async def joke(ctx):
    subreddit = reddit.subreddit("jokes")
    joke_list = []
    top = subreddit.top(limit = 25)
    for submission in top:
        joke_list.append(submission)
    random_joke = random.choice(joke_list)

    name = random_joke.title
    url = random_joke.url
    text = random_joke.selftext

    emb = discord.Embed(title = name, description = text, color = 0xf4fc58)
    await ctx.send(embed = emb)

#reddit command, runs using !reddit <sunreddit name>
@client.command(aliases=['reddit'])
async def _reddit(ctx, *, message):
    subreddit = reddit.subreddit(message)
    post_list = []
    top = subreddit.top(limit = 25)
    for submission in top:
        post_list.append(submission)
    random_post = random.choice(post_list)

    name = random_post.title
    url = random_post.url
    text = random_post.selftext
    f = ""
    if len(text.split()) > 100:
        for word in text.split()[:200]:
            f = " ".join((f, word))
        text = f + f" [continue reading...]({url})"

    emb = discord.Embed(title = name, description = text, color = 0xf4fc58)
    emb.set_image(url = url)
    await ctx.send(embed = emb)

#choose command, runs using !choose <list of options separated by 'OR'>
@client.command()
async def choose(ctx, *, message: str):
    choices = []
    choices = message.split("OR")
    answer = random.choice(choices)
    await ctx.send(answer)

#movie recommender command, runs by !movie
@client.command()
async def movie(ctx, *, message = "good"):
    movies = imdb.IMDb()
    top = movies.get_top250_movies()
    bottom = movies.get_bottom100_movies()

    if message == "good":
        movie = random.choice(top)
    else:
        movie = random.choice(bottom)

    id = movie.getID()
    movie = movies.get_movie(id)
    name = movie['title']
    directors = movie['director']
    cast = movie['cast']
    rating = movie['rating']
    dir = ', '.join(map(str, directors))
    act = ', '.join(map(str, cast[0:9]))

    title = "You should watch " + name + "!"
    text = f'**Directors:** {dir}\n\n**Cast:** {act}\n\n**Rating:** {rating}'

    emb = discord.Embed(title = title, description = text, color = 0xf4fc58)
    await ctx.send(embed = emb)

#ask for any wikipedia article, runs by !wiki <article name>
@client.command()
async def wiki(ctx, *, message):

    definition = wikipedia.summary(message, sentences=5, chars=1000,
    auto_suggest=True, redirect=True)
    title = message
    description = definition
    emb = discord.Embed(title = title, description = definition, color = 0xf4fc58)
    await ctx.send(embed = emb)

#help command: provides all details on bot functions, runs by !help
@client.command()
async def help(ctx, *, message = "all"):
    name = "Tadpole here to !help you 🐸"
    text = ''
    command_list = [
        "Type **!help** to see all my features\n\n",
        "You can type **!help <command name>** to get details for any specific command as well\n\n",
        "Type **!8ball <question>** and I will run a virtual 8ball for you!\n\n",
        "Type **!meme** and I will send you a funny meme\n\n",
        "Type **!joke** and I will tell you the best or the worst joke you have ever heard :p\n\n",
        "Type **!reddit <name of subreddit>** and I will show you a top voted post from that reddit\n\n",
        "Type **!choose <list of options separated by OR>** and I will make all your tough choices for you\n\n",
        "Type **!movie** if you want me to recommend you an awesome movie\n\n",
        "Type **!movie bad** if you want me to recommend the worst film I can think of\n\n",
        "Type **!wiki <article name>** and I will present you with an article from wikipedia\n\n",
        "You can also just start talking to me by typing **@tadpole <anything you want to say>**, I am smarter than you think!\n\n"
    ]

    if message == "all":
        for line in command_list:
            text = "".join((text, line))
    else:
        for line in command_list:
            if message in line:
                text = line

    emb = discord.Embed(title = name, description = text, color = 0xf4fc58)
    await ctx.send(embed = emb)

client.run('DISCORD BOT TOKEN') #The discord bot token key is private and should not be shared with ayone. You must create your own.
