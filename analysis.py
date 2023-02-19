import json
import os
import string
from collections import Counter, defaultdict
from datetime import datetime

import matplotlib.dates as mdates
import matplotlib.pyplot as plt
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from wordcloud import STOPWORDS, WordCloud

root_dir = "C:/Users/jblam/Downloads/messages/inbox"
stop_words = set(
    stopwords.words("english")
    + list(STOPWORDS)
    + list(string.punctuation)
    + ["'s", "'m", "'re", "n't", "'ll", "u", "im", "na"]
    + ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]
    + ["https"]
)


def parse_jsons(folder):
    messages = []
    for file in sorted(
        os.listdir(os.path.join(root_dir, folder)), reverse=True
    ):
        if os.path.splitext(file)[1] == ".json":
            msg = json.load(open(os.path.join(root_dir, folder, file)))
            msg = msg["messages"]
            msg.reverse()
            assert msg[0]["timestamp_ms"] <= msg[-1]["timestamp_ms"]
            messages += msg
    for i in range(len(messages) - 1):
        assert messages[i]["timestamp_ms"] <= messages[i + 1]["timestamp_ms"]
    return messages


def get_stats(messages):
    stats = defaultdict(
        lambda: {
            "timestamps": [],
            "messages": [],
            "message_words": [],
            "n_words_per_message": [],
            "word_counter": Counter(),
            "reacts": [],
            "reacted": [],
            "audio_files_timestamps": [],
            "files_timestamps": [],
            "videos_timestamps": [],
            "share_timestamps": [],
            "sticker_timestamps": [],
            "call_duration_timestamps": [],
            "photos_timestamps": [],
            "gifs_timestamps": [],
        }
    )
    for message in messages:
        sender = message["sender_name"]
        stats[sender]["timestamps"].append(message["timestamp_ms"])
        if "content" in message:
            if message["content"].startswith("Reacted ") and message[
                "content"
            ].endswith(" to your message "):
                continue
            if message["content"] in [
                "The video chat ended.",
                "You missed a video chat with Ambassador of fun <3.",
            ]:
                continue
            words = word_tokenize(message["content"].lower())
            stats[sender]["messages"].append(message["content"])
            stats[sender]["n_words_per_message"].append(len(words))
            words = [
                word
                for word in words
                if word.isalpha() and word not in stop_words
            ]
            stats[sender]["message_words"].append(words)
            stats[sender]["word_counter"].update(words)
        if "reactions" in message:
            for reaction in message["reactions"]:
                stats[reaction["actor"]]["reacts"].append(reaction["reaction"])
                stats[sender]["reacted"].append(reaction["reaction"])
        if "audio_files" in message:
            stats[sender]["audio_files_timestamps"].append(
                message["timestamp_ms"]
            )
        if "files" in message:
            stats[sender]["files_timestamps"].append(message["timestamp_ms"])
        if "videos" in message:
            stats[sender]["videos_timestamps"].append(message["timestamp_ms"])
        if "share" in message:
            stats[sender]["share_timestamps"].append(message["timestamp_ms"])
        if "sticker" in message:
            stats[sender]["sticker_timestamps"].append(message["timestamp_ms"])
        if "call_duration" in message:
            stats[sender]["call_duration_timestamps"].append(
                message["timestamp_ms"]
            )
        if "photos" in message:
            stats[sender]["photos_timestamps"].append(message["timestamp_ms"])
        if "gifs" in message:
            stats[sender]["gifs_timestamps"].append(message["timestamp_ms"])
    return stats


def diff_month(d1, d2):
    return (d1.year - d2.year) * 12 + d1.month - d2.month


def print_date_histogram(stats, key, date_range, name):
    fig, ax = plt.subplots(1, 1)
    for person, person_stats in stats.items():
        date_list = person_stats[key]
        date_list = [
            mdates.date2num(datetime.fromtimestamp(ms / 1000))
            for ms in date_list
        ]
        ax.hist(
            date_list,
            bins=diff_month(date_range[1], date_range[0]),
            alpha=0.5,
            label=person,
        )
        ax.xaxis.set_major_locator(mdates.YearLocator())
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%d.%m.%y"))
    ax.set_xlim([date_range[0], date_range[1]])
    plt.ylabel("Counts")
    plt.xlabel("Months")
    plt.title(name)
    plt.legend(loc="upper right")
    plt.show()


def print_wordcloud(word_lists):
    text = " ".join([" ".join(word_list) for word_list in word_lists])
    wordcloud = WordCloud(
        width=800,
        height=800,
        background_color="white",
        stopwords=stop_words,
        min_font_size=10,
    ).generate(text)

    plt.figure(figsize=(8, 8), facecolor=None)
    plt.imshow(wordcloud)
    plt.axis("off")
    plt.tight_layout(pad=0)


def print_stats(stats):
    print(
        f"Number of messages = {sum(len(person_stats['messages']) for _, person_stats in stats.items())}"
    )
    for person, person_stats in stats.items():
        print("==========", person)

        # High level stats
        avg = round(np.mean(person_stats["n_words_per_message"]), 2)
        std = round(np.std(person_stats["n_words_per_message"]), 2)
        print(f"They wrote {len(person_stats['messages'])} messages.")
        print(f"They wrote {avg} +/- {std} words per message.")
        print(f"They reacted {len(person_stats['reacts'])} times.")
        print(f"They were reacted to {len(person_stats['reacted'])} times.")

        # Word counter
        print(person_stats["word_counter"].most_common(20))
        print_wordcloud(person_stats["message_words"])

    # Dates
    first_person = list(stats.keys())[0]
    date_range = (
        datetime.fromtimestamp(
            min(stats[first_person]["timestamps"]) / 1000
        ).date(),
        datetime.fromtimestamp(
            max(stats[first_person]["timestamps"]) / 1000
        ).date(),
    )
    print(date_range)
    print_date_histogram(stats, "timestamps", date_range, "Messages")
    print_date_histogram(
        stats, "audio_files_timestamps", date_range, "Audio files"
    )
    print_date_histogram(stats, "files_timestamps", date_range, "Files")
    print_date_histogram(stats, "videos_timestamps", date_range, "Videos")
    print_date_histogram(stats, "share_timestamps", date_range, "Shares")
    print_date_histogram(stats, "sticker_timestamps", date_range, "Stickers")
    print_date_histogram(
        stats, "call_duration_timestamps", date_range, "Calls"
    )
    print_date_histogram(stats, "photos_timestamps", date_range, "Photos")
    print_date_histogram(stats, "gifs_timestamps", date_range, "GIFs")


def get_all_messages():
    all_timestamps = []
    for folder in sorted(os.listdir(root_dir)):
        folder_messages = parse_jsons(folder)
        all_timestamps += [x["timestamp_ms"] for x in folder_messages]
    print(f"Number of messages = {len(all_timestamps)}")
    date_range = (
        datetime.fromtimestamp(min(all_timestamps) / 1000).date(),
        datetime.fromtimestamp(max(all_timestamps) / 1000).date(),
    )
    print_date_histogram(
        {"JB": {"timestamps": all_timestamps}},
        "timestamps",
        date_range,
        "All of JB's messages",
    )


if __name__ == "__main__":
    messages = parse_jsons("emmyeatonkappes_10218874229560880")
    stats = get_stats(messages)
    print_stats(stats)
    # get_all_messages()
